#!/usr/bin/env python3
"""
# analyse_al_dataset.py

Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

## Usage
```
python analyse_al_dataset.py --config al_config.yaml
```

## Arguments
- `--config` (required): Path to the YAML configuration file containing the `selection_recipe` variables.
- `--top` (optional): Number of top frames to display in the CLI.

## Functionality
1. Reads `stats.json` for global run statistics including rejected frames.
2. Evaluates the `.extxyz` artifact for details regarding chosen structures.
3. Provides a clean CLI analysis report grouping selected datasets by strategy.
"""

import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from ase.io import read, iread
from ase import Atoms
from typing import List, Dict, Optional, Any

def resolve_uncertainty_array(at: Atoms, key: str) -> Optional[np.ndarray]:
    """Return the best available per-atom uncertainty array for a frame."""
    if key in at.arrays:
        return np.asarray(at.arrays[key])

    # Common fallback names used across AL generations.
    fallback_keys = ["uncertainty", "committee_uncertainty", "model_uncertainty"]
    for k in fallback_keys:
        if k in at.arrays:
            return np.asarray(at.arrays[k])
    return None


def resolve_frame_score(at, key):
    """Return a comparable scalar frame score with robust fallbacks."""
    if "al_score" in at.info:
        return float(at.info.get("al_score", 0.0))

    # max_info selectors commonly store per-frame maxima in info fields.
    info_fallback_keys = [f"max_{key}", "max_uncertainty"]
    for info_key in info_fallback_keys:
        if info_key in at.info:
            try:
                return float(at.info[info_key])
            except Exception:
                pass

    arr = resolve_uncertainty_array(at, key)
    if arr is not None and len(arr) > 0:
        return float(np.max(arr))
    return 0.0


def resolve_stress_scalar(at, stress_key):
    """Return scalar stress uncertainty for a frame (norm for vectors/tensors)."""
    if stress_key not in at.info:
        return None
    try:
        raw = np.asarray(at.info.get(stress_key), dtype=float).ravel()
    except Exception:
        return None
    if raw.size == 0:
        return None
    if raw.size == 1:
        return float(raw[0])
    return float(np.linalg.norm(raw))


def safe_percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def count_traj_frames(traj_path):
    """Count frames in an xyz trajectory without loading all of them at once."""
    try:
        return sum(1 for _ in iread(traj_path, index=":"))
    except Exception:
        return None


def make_timeline(indices, total_frames=None, width=30):
    """Build a rough ASCII timeline marking selected indices along a trajectory."""
    if width <= 0:
        return ""
    line = ["."] * width
    if not indices:
        return "".join(line)

    if total_frames is not None and total_frames > 1:
        denom = total_frames - 1
    else:
        max_idx = max(indices)
        denom = max(max_idx, 1)

    for idx in indices:
        pos = int(round((idx / denom) * (width - 1)))
        pos = max(0, min(width - 1, pos))
        line[pos] = "*"
    return "".join(line)


def print_stress_stats(selected_frames, stress_key):
    """Print stress uncertainty table using frame-level stress info."""
    values = []
    per_strategy = defaultdict(list)

    for at in selected_frames:
        sval = resolve_stress_scalar(at, stress_key)
        if sval is None:
            continue
        values.append(sval)
        per_strategy[at.info.get("al_strategy", "unknown")].append(sval)

    print_header("STRESS UNCERTAINTY (FRAME LEVEL)")
    if not values:
        print(f"[WARN] No '{stress_key}' values found in frame info.")
        return

    header = f"{'Group':<30} | {'Count':<8} | {'Mean':<10} | {'Max':<10} | {'95th %':<10}"
    print(header)
    print("-" * len(header))
    print(
        f"{'All':<30} | {len(values):<8} | {np.mean(values):<10.4f} | "
        f"{np.max(values):<10.4f} | {safe_percentile(values, 95):<10.4f}"
    )

    for strategy, vals in sorted(per_strategy.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(
            f"{strategy:<30} | {len(vals):<8} | {np.mean(vals):<10.4f} | "
            f"{np.max(vals):<10.4f} | {safe_percentile(vals, 95):<10.4f}"
        )


def print_pick_position_summary(selected_frames, runs_dir, timeline_width=30):
    """Show rough locations in each trajectory where frames were selected."""
    run_to_indices = defaultdict(list)
    for at in selected_frames:
        run = at.info.get("src_run", "unknown")
        idx = at.info.get("src_index", None)
        try:
            run_to_indices[run].append(int(idx))
        except Exception:
            continue

    print_header("WHERE FRAMES WERE PICKED IN TRAJECTORY")
    if not run_to_indices:
        print("[WARN] No src_run/src_index metadata found for timeline summary.")
        return

    runs_dir_path = Path(runs_dir) if runs_dir else None
    traj_len_cache = {}

    header = (
        f"{'Run':<15} | {'Picks':<6} | {'MinIdx':<7} | {'MaxIdx':<7} | "
        f"{'MeanIdx':<8} | {'Span%':<7} | {'Timeline':<{timeline_width}}"
    )
    print(header)
    print("-" * len(header))

    for run in sorted(run_to_indices.keys()):
        idxs = sorted(run_to_indices[run])
        total_frames = None

        if runs_dir_path is not None and run != "unknown":
            traj_path = runs_dir_path / run / "traj.xyz"
            if run not in traj_len_cache:
                traj_len_cache[run] = count_traj_frames(traj_path) if traj_path.exists() else None
            total_frames = traj_len_cache[run]

        min_idx = min(idxs)
        max_idx = max(idxs)
        mean_idx = float(np.mean(idxs))
        if total_frames is not None and total_frames > 1:
            span_pct = 100.0 * (max_idx - min_idx) / (total_frames - 1)
            span_pct_txt = f"{span_pct:.1f}"
        else:
            span_pct_txt = "N/A"

        timeline = make_timeline(idxs, total_frames=total_frames, width=timeline_width)
        print(
            f"{run:<15} | {len(idxs):<6} | {min_idx:<7} | {max_idx:<7} | "
            f"{mean_idx:<8.1f} | {span_pct_txt:<7} | {timeline:<{timeline_width}}"
        )

def print_header(title):
    print(f"\n--- {title} ---")

def load_manifest(json_path):
    if not Path(json_path).exists():
        print(f"[ERROR] Manifest file not found: {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not parse manifest: {e}")
        return None

def print_global_stats(manifest, key):
    """Prints stats for ALL frames (rejected + selected) from the JSON manifest."""
    if key not in manifest:
        print(f"\n[WARN] Key '{key}' not found in manifest. Available keys: {list(manifest.keys())}")
        return

    entry = manifest[key]
    stats = entry.get("global_stats", {})

    print_header(f"GLOBAL STATISTICS (ALL FRAMES) - Key: {key}")
    header = f"{'Species':<8} | {'Mean':<10} | {'Max':<10} | {'95th %':<10} | {'99th %':<10} | {'Count':<10}"
    sep = "-" * len(header)

    print("\n" + header)
    print(sep)

    if "All" in stats:
        data = stats["All"]
        mean_val = data.get("mean", 0.0)
        max_val = data.get("max", 0.0)
        p95 = data.get("p95", 0.0)
        p99 = data.get("p99", 0.0)
        count = data.get("count", 0)

        print(f"{'All':<8} | {mean_val:<10.4f} | {max_val:<10.4f} | {p95:<10.4f} | {p99:<10.4f} | {count:<10}")
        print(sep)

    # Print remaining species (excluding "All") in sorted order
    for sym, data in sorted((k, v) for k, v in stats.items() if k != "All"):
        mean_val = data.get("mean", 0.0)
        max_val = data.get("max", 0.0)
        p95 = data.get("p95", 0.0)
        p99 = data.get("p99", 0.0)
        count = data.get("count", 0)

        print(f"{sym:<8} | {mean_val:<10.4f} | {max_val:<10.4f} | {p95:<10.4f} | {p99:<10.4f} | {count:<10}")

def print_artifact_stats(xyz_path, key, top_n_show=15, stress_key="stress_uncertainty", runs_dir="./active_learning_runs", timeline_width=30):
    """Analyses the actual atoms selected for training from the .extxyz file."""
    path = Path(xyz_path)
    if not path.exists():
        print(f"\n[WARN] Artifact file not found: {path}")
        return

    print(f"\nLoading selected frames from {path}...")
    try:
        # Load all frames
        frames = read(path, index=":")
        if not isinstance(frames, list): frames = [frames]
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return

    # Filter for frames selected by this specific key
    # (In case the file contains multiple selection batches)
    selected_frames = [at for at in frames if at.info.get("al_key") == key]

    # Fallback for runs where al_key is absent or uses older naming.
    if not selected_frames and key == "uncertainty":
        selected_frames = [
            at for at in frames
            if at.info.get("al_key") in (None, "", "max_uncertainty", "committee_uncertainty")
        ]

    if not selected_frames:
        print(f"[INFO] No frames found in {path} with al_key='{key}'.")
        # Fallback: if user didn't write al_key in older versions or mixed usage
        print("       (Listing all frames instead, assuming single-batch file)")
        selected_frames = frames

    print_header(f"SELECTED DATASET STATS (TRAINING SET) - {len(selected_frames)} Frames")

    # 1. Strategy Breakdown
    strategies = Counter([at.info.get("al_strategy", "unknown") for at in selected_frames])
    print(f"{'Strategy':<30} | {'Count':<10}")
    print("-" * 45)
    for s, c in strategies.most_common():
        print(f"{s:<30} | {c:<10}")

    # 1b. Optional provenance split by run status (completed/failed/unknown).
    statuses = Counter([at.info.get("src_status", "unknown") for at in selected_frames])
    if statuses:
        print("\nStatus breakdown:")
        print(f"{'Status':<20} | {'Count':<10}")
        print("-" * 33)
        for s, c in statuses.most_common():
            print(f"{s:<20} | {c:<10}")

    # 2. Uncertainty Stats of Selected Frames
    species_data = defaultdict(list)

    # Check if the array exists
    found_array = False
    for at in selected_frames:
        unc = resolve_uncertainty_array(at, key)
        if unc is not None:
            found_array = True
            syms = at.get_chemical_symbols()
            for s, u in zip(syms, unc):
                species_data[s].append(u)

    if found_array:
        print("\nDistribution of uncertainty within selected frames:")
        header = f"{'Species':<8} | {'Mean':<10} | {'Max':<10} | {'99th %':<10} | {'Count':<10}"
        print(header)
        print("-" * len(header))

        # Global stats (All species)
        all_unc = []
        for vals in species_data.values():
            all_unc.extend(vals)
        if all_unc:
            arr_all = np.array(all_unc)
            print(f"{'All':<8} | {np.mean(arr_all):<10.4f} | {np.max(arr_all):<10.4f} | {np.percentile(arr_all, 99):<10.4f} | {len(arr_all):<10}")
            print("-" * len(header))

        for sym in sorted(species_data.keys()):
            arr = np.array(species_data[sym])
            print(f"{sym:<8} | {np.mean(arr):<10.4f} | {np.max(arr):<10.4f} | {np.percentile(arr, 99):<10.4f} | {len(arr):<10}")
    else:
        print(f"\n[WARN] No compatible uncertainty array found for key '{key}'. Cannot calculate local stats.")

    # 2b. Stress uncertainty summary
    print_stress_stats(selected_frames, stress_key)

    # 2c. Rough position of selected indices in each trajectory
    print_pick_position_summary(selected_frames, runs_dir, timeline_width=timeline_width)

    # 3. Top Uncertainty Frames
    print_top_frames(selected_frames, key, n=top_n_show)

def print_top_frames(frames, key, n=15):
    print_header(f"TOP {n} HIGHEST SCORED FRAMES")

    # Sort by stored score
    ranked = sorted(frames, key=lambda x: resolve_frame_score(x, key), reverse=True)
    top = ranked[:n]

    header = f"{'Run':<15} | {'Frame':<6} | {'Strategy':<25} | {'Score':<10} | {'MaxSpecies':<15}"
    print(header)
    print("-" * len(header))

    for at in top:
        run = at.info.get("src_run", "unknown")
        idx = at.info.get("src_index", "N/A")
        strategy = at.info.get("al_strategy", "unknown")
        score = resolve_frame_score(at, key)

        max_species = "N/A"
        unc = resolve_uncertainty_array(at, key)
        if unc is not None:
            syms = np.asarray(at.get_chemical_symbols())

            if len(unc) == len(syms) and len(unc) > 0:
                max_u = np.max(unc)
                # indices where uncertainty equals the max (exact match)
                max_idx = np.where(unc == max_u)[0]
                # unique species among the max atoms
                max_species = ",".join(sorted(set(syms[max_idx].tolist())))

        print(f"{run:<15} | {idx:<6} | {strategy:<25} | {score:<10.4f} | {max_species:<15}")

def main() -> None:
    """
    Parses configuration and initializes the CLI analytics.
    """
    parser = argparse.ArgumentParser(description="Active Learning Dataset Analyzer.")
    parser.add_argument("--config", type=str, required=True, help="Path to the active learning YAML configuration.")
    parser.add_argument("--top", type=int, default=15, help="Number of top frames to display.")
    parser.add_argument("--timeline-width", type=int, default=30, help="Width of per-run ASCII timeline.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    cfg = full_config.get("selection_recipe", full_config)
    committee_cfg = full_config.get("committee_params", {})
    
    key = cfg.get("uncertainty_key", "uncertainty")
    stress_key = cfg.get("stress_key", "stress_uncertainty")
    stats_json = committee_cfg.get("output_json", "stats.json")
    xyz_path = cfg.get("output_extxyz", "selected_frames.extxyz")
    runs_dir = cfg.get("runs_dir", "./active_learning_runs")

    manifest = load_manifest(stats_json)

    if manifest:
        print_global_stats(manifest, key)
        entry = manifest.get(key, {})
        if entry.get("xyz_file"):
             xyz_path = entry.get("xyz_file")
    else:
        print("\n[INFO] Proceeding with artifact-only analysis (manifest unavailable).")

    print_artifact_stats(
        xyz_path,
        key,
        top_n_show=args.top,
        stress_key=stress_key,
        runs_dir=runs_dir,
        timeline_width=args.timeline_width,
    )

if __name__ == "__main__":
    main()
