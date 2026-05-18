#!/usr/bin/env python3
"""
# run_committee.py

Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

## Usage
```
python run_committee.py --config al_config.yaml
```

## Arguments
- `--config` (required): Path to the YAML configuration file containing the `committee_params` block.

## Functionality
1. Loads an ensemble of ML potentials specified in the configuration yaml.
2. Iterates over frames from MD trajectories ensuring robust evaluation.
3. Computes the variance between model predictions (force and stress).
4. Saves evaluated arrays back into the `.xyz` trajectories.
5. Populates a global JSON manifest (`stats.json`) for data analysis.
"""

import argparse
import json
import logging
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from ase.io import read, write
from nequip.ase import NequIPCalculator

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(handler)

class StatsTracker:
    def __init__(self, uncertainty_key):
        self.data = defaultdict(list)
        self.total_frames = 0
        self.uncertainty_key = uncertainty_key

    def update(self, atoms):
        self.total_frames += 1
        if self.uncertainty_key not in atoms.arrays: return

        unc = atoms.arrays[self.uncertainty_key]
        syms = atoms.get_chemical_symbols()

        for s, u in zip(syms, unc):
            self.data[s].append(u)

    def get_report(self):
        report = {}
        all_values = []

        for sym, values in sorted(self.data.items()):
            all_values.extend(values)
            arr = np.array(values)
            stats = {
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "count": len(arr)
            }
            report[sym] = stats

        if all_values:
            arr_all = np.array(all_values)
            report["All"] = {
                "mean": float(np.mean(arr_all)),
                "max": float(np.max(arr_all)),
                "p95": float(np.percentile(arr_all, 95)),
                "p99": float(np.percentile(arr_all, 99)),
                "count": len(arr_all)
            }
        return report

class CommitteeEnsemble:
    def __init__(self, model_paths, device, use_base_forces=False):
        self.calculators = []
        self.use_base_forces = use_base_forces
        if model_paths:
            for path in tqdm(model_paths, desc="Loading Models", leave=False):
                self.calculators.append(NequIPCalculator.from_compiled_model(compile_path=path, device=device))
            print(f"{len(self.calculators)} Models loaded")

    def compute_uncertainty(self, atoms):
        forces_stack = []
        stress_stack = []

        if self.use_base_forces:
            # Try to grab forces from existing calculator or array
            f = None
            s = None
            if atoms.calc:
                try:
                    f = atoms.get_forces()
                    s = atoms.get_stress()
                except Exception: pass

            if f is None and "forces" in atoms.arrays:
                f = atoms.arrays["forces"]
                
            if s is None and "stress" in atoms.info:
                s = atoms.info["stress"]

            if f is not None:
                forces_stack.append(f)
            if s is not None:
                stress_stack.append(s)

        original_calc = getattr(atoms, "calc", None)

        for calc in self.calculators:
            atoms.calc = calc
            forces_stack.append(atoms.get_forces())
            stress_stack.append(atoms.get_stress())

        atoms.calc = original_calc

        if not forces_stack:
            return np.zeros(len(atoms)), np.zeros(6), np.zeros(6)

        forces_unc = np.linalg.norm(np.std(np.array(forces_stack), axis=0), axis=1)
        stress_arr = np.array(stress_stack)
        stress_unc = np.std(stress_arr, axis=0)
        mean_stress = np.mean(stress_arr, axis=0)

        return forces_unc, stress_unc, mean_stress

def process_run(run_dir, committee, stats, cfg):
    traj_path = run_dir / "traj.xyz"
    crash_path = run_dir / "crash_dump.xyz"

    frames = []

    # 1. Load existing frames
    if traj_path.exists():
        try:
            loaded = read(traj_path, index=":")
            if not isinstance(loaded, list):
                loaded = [loaded]

            ukey = cfg.get("uncertainty_key", "uncertainty")
            if not cfg["overwrite"] and len(loaded) > 0 and ukey in loaded[0].arrays:
                for at in tqdm(
                    loaded,
                    desc=f"{run_dir.name} stats",
                    unit="frame",
                    leave=False,
                    position=1,
                ):
                    stats.update(at)
                return

            frames.extend(loaded)
        except Exception:
            pass

    if crash_path.exists():
        try:
            loaded = read(crash_path, index=":")
            if not isinstance(loaded, list):
                loaded = [loaded]
            frames.extend(loaded)
        except Exception:
            pass

    if not frames:
        return

    # 2. Process Frames
    valid_frames = []
    ukey = cfg.get("uncertainty_key", "uncertainty")

    for i, at in enumerate(
        tqdm(
            frames,
            desc=f"{run_dir.name} frames",
            unit="frame",
            leave=False,
            position=1,
        )
    ):
        unc, stress_unc, mean_stress = committee.compute_uncertainty(at)
        at.new_array(ukey, unc)

        if getattr(at, "calc", None) is not None and "stress" in at.calc.results:
            at.calc.results["stress"] = mean_stress
        else:
            at.info["stress"] = mean_stress

        at.info[f"stress_{ukey}"] = stress_unc
        at.info["src_index"] = i
        at.info[f"max_{ukey}"] = float(np.max(unc))
        at.info[f"mean_{ukey}"] = float(np.mean(unc))

        valid_frames.append(at)
        stats.update(at)

    # 3. Overwrite traj.xyz
    if valid_frames:
        temp_path = run_dir / "traj.temp.xyz"
        write(temp_path, valid_frames)
        temp_path.replace(traj_path)
        logger.info(f"Processed {run_dir.name}: {len(valid_frames)} frames")

def main() -> None:
    """
    Parses configuration and initializes the committee uncertainty calculation.
    """
    parser = argparse.ArgumentParser(description="Active Learning Committee Evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the active learning YAML configuration.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    cfg = full_config.get("committee_params", full_config)

    ukey = cfg.get("uncertainty_key", "uncertainty")
    stats = StatsTracker(ukey)
    json_path = Path(cfg.get("output_json", "stats.json"))

    committee = CommitteeEnsemble(cfg["models"], cfg["device"], use_base_forces=cfg.get("use_base_forces", False))
    base_path = Path(cfg["runs_dir"])
    runs = sorted([p for p in base_path.glob("run_*") if p.is_dir()],
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 999999)

    # Main Progress Bar
    for run in tqdm(
        runs,
        desc="Computing Uncertainties",
        unit="run",
        position=0,
    ):
        process_run(run, committee, stats, cfg)

    # Save Manifest
    manifest = {}
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                manifest = json.load(f)
        except Exception: pass

    manifest[ukey] = {
        "global_stats": stats.get_report(),
        "total_frames_seen": stats.total_frames
    }

    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated global stats in {json_path}")

if __name__ == "__main__":
    main()
