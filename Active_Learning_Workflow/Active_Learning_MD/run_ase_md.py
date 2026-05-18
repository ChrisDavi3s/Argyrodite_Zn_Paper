#!/usr/bin/env python3
"""
# run_ase_md.py

Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

## Usage
```
python run_ase_md.py --config al_config.yaml
```

## Arguments
- `--config` (required): Path to the YAML configuration file containing the `md_params` block.

## Functionality
1. Loads MD configuration from the specified YAML file.
2. Runs NVT or NPT Molecular Dynamics using ASE and NequIP potentials.
3. Automatically caches buffer frames and writes 'crash dumps' if limits are exceeded.
4. Manages stateful resumes for interrupted batch evaluations.
"""

import argparse
import json
import os
from collections import deque
from typing import Dict, Any, List

import yaml
import numpy as np
import torch
from ase import Atoms, units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.constraints import FixAtoms
from nequip.ase import NequIPCalculator
from tqdm import tqdm
from ase.filters import UnitCellFilter
from ase.optimize import FIRE

def get_run_state(run_dir: str) -> str:
    """
    Returns: one of
    - "missing"     : run dir does not exist
    - "abandoned"   : run dir exists but no valid status.json
    - "completed"
    - "failed"
    - "crashed"
    """
    if not os.path.exists(run_dir):
        return "missing"

    status_path = os.path.join(run_dir, "status.json")
    if not os.path.exists(status_path):
        return "abandoned"

    try:
        with open(status_path) as f:
            status = json.load(f).get("status", None)
        if status in {"completed", "failed", "crashed"}:
            return status
        return "abandoned"
    except Exception:
        return "abandoned"

def pre_relax(atoms, calc, cfg, index=None):
    """
    Cell + position relax.
    Uses cell filter with to enable isotropic strain.
    Raises RuntimeError on failure.
    """
    atoms.calc = calc

    ucf = UnitCellFilter(
        atoms,
        hydrostatic_strain=cfg.get("relax_isotropic", True)
    )

    opt = FIRE(ucf, logfile=None)

    pbar = tqdm(
        range(cfg["relax_steps"]),
        desc=f"Relax {index}",
        leave=False,
        dynamic_ncols=True,
    )

    try:
        for step in pbar:
            opt.step()

            forces = atoms.get_forces()
            fmax = np.linalg.norm(forces, axis=1).max()

            pbar.set_postfix({"Fmax": f"{fmax:.3f}"})

            if fmax < cfg["relax_fmax"]:
                return

            if fmax > cfg["max_force_ev_a"]:
                raise RuntimeError(f"Relax force limit exceeded: {fmax:.2f} > {cfg['max_force_ev_a']}")

    except Exception as e:
        raise RuntimeError(f"Relax crashed: {e}")

    raise RuntimeError(
        f"Relax did not converge: fmax > {cfg['relax_fmax']} "
        f"after {cfg['relax_steps']} steps"
    )

def run_simulation(atoms, calc, index, cfg):
    run_dir = os.path.join(cfg["output_dir"], f"run_{index}")
    os.makedirs(run_dir, exist_ok=True)
    traj_path = os.path.join(run_dir, "traj.xyz")
    crash_path = os.path.join(run_dir, "crash_dump.xyz")
    log_path = os.path.join(run_dir, "status.json")
    crash_dumped = False

    if os.path.exists(traj_path):
        os.remove(traj_path)

    atoms.calc = calc

    if cfg.get("relax", False):
        try:
            pre_relax(atoms, calc, cfg, index=index)
        except Exception as e:
            status = "failed"
            fail_msg = f"Pre-relax failed: {e}"

            summary = {
                "index": index,
                "status": status,
                "steps": 0,
                "max_force": 0.0,
                "final_T": 0.0,
                "fail_reason": fail_msg,
                "crash_dumped": False
            }

            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)

            return summary

    MaxwellBoltzmannDistribution(atoms, temperature_K=cfg["temp_k"])
    Stationary(atoms)
    ZeroRotation(atoms)

    # Integrator Selection
    timestep = cfg["timestep_fs"] * units.fs

    if cfg["ensemble"].lower() == "npt":

        if cfg.get("isotropic", False):
            dyn = IsotropicMTKNPT(
                atoms=atoms,
                timestep=timestep,
                temperature_K=cfg["temp_k"],
                pressure_au=cfg["pressure_bar"] * units.bar,
                tdamp=cfg["t_damp_fs"] * units.fs,
                pdamp=cfg["p_damp_fs"] * units.fs,
            )
        else:
            dyn = MTKNPT(
                atoms=atoms,
                timestep=timestep,
                temperature_K=cfg["temp_k"],
                pressure_au=cfg["pressure_bar"] * units.bar,
                tdamp=cfg["t_damp_fs"] * units.fs,
                pdamp=cfg["p_damp_fs"] * units.fs,
            )
    else:
        dyn = Langevin(
            atoms=atoms,
            timestep=timestep,
            temperature_K=cfg["temp_k"],
            friction=cfg["friction"],
        )

    # Numpy Ring Buffer
    # Instead of copying Atoms objects, we pre-allocate numpy arrays.
    buf_size = cfg["crash_buffer_size"]
    n_atoms = len(atoms)

    # Arrays to store history
    buf_pos = np.zeros((buf_size, n_atoms, 3))
    buf_forces = np.zeros((buf_size, n_atoms, 3))
    buf_cells = np.zeros((buf_size, 3, 3))

    status = "completed"
    fail_msg = "None"
    max_f_seen = 0.0
    steps_run = 0

    pbar = tqdm(
        range(cfg["steps"]), desc=f"ID {index}", leave=False, dynamic_ncols=True
    )

    for step in pbar:
        dyn.run(1)
        steps_run += 1

        # 1. Get Data
        # atoms.get_forces() is cached by the calculator, so calling it here is cheap
        forces = atoms.get_forces()
        pos = atoms.get_positions()
        cell = atoms.get_cell().array

        fmax = np.linalg.norm(forces, axis=1).max()
        max_f_seen = max(max_f_seen, fmax)

        # 2. Update Ring Buffer
        # We use modulo operator % to wrap around the array
        buf_idx = (steps_run - 1) % buf_size
        buf_pos[buf_idx] = pos
        buf_forces[buf_idx] = forces
        buf_cells[buf_idx] = cell

        # 3. Stability Check
        if fmax > cfg["max_force_ev_a"]:
            status = "failed"
            fail_msg = f"Force limit exceeded: {fmax:.2f} > {cfg['max_force_ev_a']}"

            crash_frames = []

            n_frames_available = min(steps_run, buf_size)

            if steps_run >= buf_size:
                start = (buf_idx + 1) % buf_size  # oldest frame
            else:
                start = 0

            for i in range(n_frames_available):
                read_idx = (start + i) % buf_size

                frame = atoms.copy()
                frame.set_positions(buf_pos[read_idx])
                frame.set_cell(buf_cells[read_idx])
                frame.arrays["forces"] = buf_forces[read_idx]

                crash_frames.append(frame)

            write(crash_path, crash_frames)
            crash_dumped = True
            break

        # I/O (Trajectory)
        if step % cfg["dump_freq"] == 0:
            write(traj_path, atoms, append=True)

        if step % 10 == 0:
            pbar.set_postfix(
                {"T": f"{atoms.get_temperature():.0f}K", "Fmax": f"{fmax:.2f}"}
            )

    # --- Save Log ---
    summary = {
        "index": index,
        "status": status,
        "steps": steps_run,
        "max_force": float(max_f_seen),
        "final_T": float(atoms.get_temperature()),
        "fail_reason": fail_msg,
        "crash_dumped": crash_dumped,
    }

    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """
    Parses configuration and initializes the batch MD pipeline.
    """
    parser = argparse.ArgumentParser(description="Batch MD wrapper for ASE/NequIP.")
    parser.add_argument("--config", type=str, required=True, help="Path to the active learning YAML configuration.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    cfg = full_config.get("md_params", full_config)

    # 1. Global Setup
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Save MD Parameters for reproducibility
    with open(os.path.join(cfg["output_dir"], "md_params.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Loading model: {cfg['model_path']} ({cfg['device']})")

    calc = NequIPCalculator.from_compiled_model(
        compile_path=cfg["model_path"],
        device=cfg["device"],
        chemical_species_to_atom_type_map=True,
    )

    print(f"Reading inputs: {cfg['input_file']}")
    structures = read(cfg["input_file"], index=":")
    # process constraints and clear existing
    for s in structures:
        s.set_calculator(None)
        
        if "freeze_z_pct" in cfg and cfg["freeze_z_pct"] > 0:
            if cfg.get("ensemble", "nvt").lower() == "npt":
                print(f"WARNING: freeze_z_pct is set to {cfg['freeze_z_pct']}, but ensemble is NPT. Skipping freezing to avoid unphysical stresses.")
                s.set_constraint(None)
            else:
                pct = cfg["freeze_z_pct"]
                z_coords = s.positions[:, 2]
                z_min = np.min(z_coords)
                z_max = np.max(z_coords)
                z_range = z_max - z_min
                
                # Bottom and Top percentages by coordinates
                bottom_thresh = z_min + z_range * pct
                top_thresh = z_max - z_range * pct
                
                # Identify indices to freeze
                mask = (z_coords < bottom_thresh) | (z_coords > top_thresh)
                indices = np.where(mask)[0]
                
                s.set_constraint(FixAtoms(indices=indices))
                print(f"Freezing {len(indices)} of {len(s)} atoms (bottom and top {pct*100}%)")
        else:
            s.set_constraint(None)

    results = []

    # 2. Batch Execution
    print(f"Starting {len(structures)} runs in '{cfg['output_dir']}'...")

    # Outer progress bar for the batch
    for i, atoms in enumerate(tqdm(structures, desc="Batch Progress")):
        run_dir = os.path.join(cfg["output_dir"], f"run_{i}")
        log_path = os.path.join(run_dir, "status.json")
        state = get_run_state(run_dir)

        # Helper to load existing log for summary
        def load_existing_result():
            try:
                with open(log_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {
                    "index": i,
                    "status": state,
                    "steps": 0,
                    "max_force": 0.0,
                    "fail_reason": "Log file missing/corrupt"
                }

        if state == "completed":
            tqdm.write(f"Run {i} SKIPPED (completed)")
            results.append(load_existing_result())
            continue

        if state in {"failed", "crashed"}:
            tqdm.write(f"Run {i} SKIPPED ({state})")
            results.append(load_existing_result())
            continue

        if state == "abandoned":
            if not cfg["resume"]:
                tqdm.write(f"Run {i} SKIPPED (abandoned, resume=False)")
                results.append(load_existing_result())
                continue
            tqdm.write(f"Run {i} RESUMING (abandoned)")

        # missing OR abandoned with resume=True means run
        try:
            res = run_simulation(atoms, calc, i, cfg)
            results.append(res)

            if res["status"] == "completed":
                tqdm.write(
                    f"Run {i} COMPLETED: {res['steps']} steps, "
                    f"Max F: {res['max_force']:.2f}"
                )

            elif res["status"] == "failed":
                msg = (
                    f"Run {i} FAILED: Step {res['steps']}/{cfg['steps']}: "
                    f"{res['fail_reason']}"
                )
                if res.get("crash_dumped", False):
                    msg += f" (Dumped last {cfg['crash_buffer_size']} frames)"
                tqdm.write(msg)

            else:
                tqdm.write(f"Run {i} ended with status: {res['status']}")

        except Exception as e:
            err_msg = str(e)
            tqdm.write(f"Run {i} CRASHED: {err_msg}")

            # Log crash details
            crash_dir = os.path.join(cfg["output_dir"], f"run_{i}")
            os.makedirs(crash_dir, exist_ok=True)
            with open(os.path.join(crash_dir, "CRASH_LOG.txt"), "w") as f:
                f.write(err_msg)

            results.append(
                {
                    "index": i,
                    "status": "crashed",
                    "steps": 0,
                    "max_force": 0.0,
                    "final_T": 0.0,
                    "fail_reason": err_msg,
                }
            )

    # 3. Final Summary & Statistics
    total_runs = len(results)
    if total_runs == 0:
        print("No results to summarise.")
        return

    # Count statuses
    counts = {"completed": 0, "failed": 0, "crashed": 0, "abandoned": 0, "missing": 0}
    for r in results:
        s = r.get("status", "missing")
        counts[s] = counts.get(s, 0) + 1

    summary_lines = []
    header = f"{'ID':<5} | {'Status':<10} | {'Steps':<8} | {'MaxF':<8} | {'Reason'}"
    sep = "-" * len(header)

    summary_lines.append("--- BATCH SUMMARY ---")
    summary_lines.append(header)
    summary_lines.append(sep)

    for r in results:
        # Handle cases where keys might be missing in old logs
        idx = r.get("index", "?")
        st = r.get("status", "unknown")
        stp = r.get("steps", 0)
        mf = r.get("max_force", 0.0)
        rsn = r.get("fail_reason", "")

        line = f"{idx:<5} | {st:<10} | {stp:<8} | {mf:<8.2f} | {rsn}"
        summary_lines.append(line)

    # Add Statistics Section
    summary_lines.append("\n--- STATISTICS ---")
    summary_lines.append(f"Total Runs: {total_runs}")

    for key in ["completed", "failed", "crashed"]:
        count = counts.get(key, 0)
        pct = (count / total_runs) * 100
        summary_lines.append(f"{key.capitalize():<10}: {count:>4} ({pct:>5.1f}%)")

    summary_text = "\n".join(summary_lines)

    # Save to file
    with open(os.path.join(cfg["output_dir"], "batch_summary.txt"), "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)

if __name__ == "__main__":
    main()
