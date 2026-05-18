"""
Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

select_frames.py
Selects active learning candidates for re-training with a flexible multi-selector API.

Usage:
    python select_frames.py --config al_config.yaml

Outputs:
Artifact (.extxyz): The actual selected frames with provenance tags.
"""

import argparse
import json
import random
import yaml
import numpy as np
from heapq import nlargest, nsmallest
from pathlib import Path
from ase.io import read, write
from tqdm import tqdm


def get_run_status(run_dir):
    status_file = run_dir / "status.json"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f).get("status", "unknown")
        except Exception:
            pass
    return "unknown"


def get_min_dist(atoms, species_pair):
    """Calculates the minimum distance between two species in the given atoms object."""
    try:
        dist_mat = atoms.get_all_distances(mic=True)
    except Exception:
        # Fallback to non-MIC if unit cell is weird
        dist_mat = atoms.get_all_distances(mic=False)

    sym = np.array(atoms.get_chemical_symbols())
    idx1 = np.where(sym == species_pair[0])[0]
    idx2 = np.where(sym == species_pair[1])[0]

    if len(idx1) == 0 or len(idx2) == 0:
        return float("inf")

    sub_dist = dist_mat[np.ix_(idx1, idx2)].astype(float)

    if species_pair[0] == species_pair[1]:
        # Ignore self-distances on the diagonal
        np.fill_diagonal(sub_dist, np.inf)

    return np.min(sub_dist)


def apply_filters(frames_with_idx, filters_cfg, min_dist_cache=None):
    """
    Applies global rejection criteria.

    frames_with_idx is a list of tuples:
        (original_index, atoms)
    """
    valid = []

    for idx, at in frames_with_idx:
        passed = True

        for fcfg in filters_cfg:
            if fcfg["type"] == "min_distance":
                species = tuple(fcfg["species"])
                cache_key = (idx, species)

                if min_dist_cache is not None and cache_key in min_dist_cache:
                    min_d = min_dist_cache[cache_key]
                else:
                    min_d = get_min_dist(at, species)
                    if min_dist_cache is not None:
                        min_dist_cache[cache_key] = min_d

                if min_d < fcfg["threshold"]:
                    passed = False
                    break

        if passed:
            valid.append((idx, at))

    return valid


def is_spaced_from_selected(idx, selected_indices, spacing):
    """Returns True if idx is more than spacing away from every selected index."""
    if spacing <= 0:
        return True

    return all(abs(idx - selected_idx) > spacing for selected_idx in selected_indices)


def apply_spacing_to_ranked_candidates(ranked_candidates, n, spacing, existing_selected_indices=None):
    """
    Greedily chooses up to n ranked candidates while enforcing frame-index spacing.

    This matters inside one selector. For example, if max_info asks for n=64,
    this prevents those 64 frames from being adjacent.
    """
    if existing_selected_indices is None:
        existing_selected_indices = []

    if spacing <= 0:
        return ranked_candidates[:n]

    picked = []
    blocked_indices = list(existing_selected_indices)

    for idx, at in ranked_candidates:
        if is_spaced_from_selected(idx, blocked_indices, spacing):
            picked.append((idx, at))
            blocked_indices.append(idx)

        if len(picked) >= n:
            break

    return picked


def execute_selector(
    scfg,
    candidates,
    score_cache=None,
    distance_compute_enabled=True,
    spacing=0,
    existing_selected_indices=None,
):
    """Returns a list of selected frames: (idx, atoms, reason), up to limit n."""
    stype = scfg["type"]
    n = scfg.get("n", 1)
    score_cache = score_cache or {}

    if existing_selected_indices is None:
        existing_selected_indices = []

    if not candidates or n <= 0:
        return []

    # Allow selector-specific spacing to override global spacing.
    spacing = int(scfg.get("spacing", spacing))

    if stype == "max_info":
        key = scfg["key"]

        ranked = nlargest(
            len(candidates),
            candidates,
            key=lambda x: x[1].info.get(key, -1e9),
        )

        picked = apply_spacing_to_ranked_candidates(
            ranked,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, f"{stype}_{key}") for idx, at in picked]

    elif stype == "max_stress_info":
        key = scfg["key"]
        cache = score_cache.setdefault("max_stress_info", {})

        def stress_norm(at):
            cache_key = (id(at), key)
            if cache_key in cache:
                return cache[cache_key]

            val = at.info.get(key)
            if val is not None:
                score = float(np.linalg.norm(val))
            else:
                score = -1e9

            cache[cache_key] = score
            return score

        ranked = nlargest(
            len(candidates),
            candidates,
            key=lambda x: stress_norm(x[1]),
        )

        picked = apply_spacing_to_ranked_candidates(
            ranked,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, f"{stype}_{key}") for idx, at in picked]

    elif stype == "closest_distance":
        if not distance_compute_enabled:
            return []

        species = tuple(scfg["species"])
        cache = score_cache.setdefault("min_distance", {})

        def min_dist_for_candidate(candidate):
            idx, at = candidate
            cache_key = (idx, species)

            if cache_key not in cache:
                cache[cache_key] = get_min_dist(at, species)

            return cache[cache_key]

        ranked = nsmallest(
            len(candidates),
            candidates,
            key=min_dist_for_candidate,
        )

        picked = apply_spacing_to_ranked_candidates(
            ranked,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, f"{stype}_{species[0]}-{species[1]}") for idx, at in picked]

    elif stype == "furthest_distance":
        if not distance_compute_enabled:
            return []

        species = tuple(scfg["species"])
        cache = score_cache.setdefault("min_distance", {})

        def min_dist_for_candidate(candidate):
            idx, at = candidate
            cache_key = (idx, species)

            if cache_key not in cache:
                cache[cache_key] = get_min_dist(at, species)

            return cache[cache_key]

        ranked = nlargest(
            len(candidates),
            candidates,
            key=min_dist_for_candidate,
        )

        picked = apply_spacing_to_ranked_candidates(
            ranked,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, f"{stype}_{species[0]}-{species[1]}") for idx, at in picked]

    elif stype == "max_force":
        cache = score_cache.setdefault("max_force", {})

        def get_max_f(at):
            cache_key = id(at)

            if cache_key in cache:
                return cache[cache_key]

            if at.calc:
                f = at.get_forces()
            else:
                f = at.arrays.get("forces", np.zeros((len(at), 3)))

            score = float(np.linalg.norm(f, axis=1).max())
            cache[cache_key] = score
            return score

        ranked = nlargest(
            len(candidates),
            candidates,
            key=lambda x: get_max_f(x[1]),
        )

        picked = apply_spacing_to_ranked_candidates(
            ranked,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, stype) for idx, at in picked]

    elif stype == "last_valid":
        offset = int(scfg.get("offset", 0))
        force_cap = scfg.get("max_force", None)

        valid = []
        for idx, at in candidates:
            if force_cap is not None:
                if at.calc:
                    f = at.get_forces()
                else:
                    f = at.arrays.get("forces", np.zeros((len(at), 3)))
                if float(np.linalg.norm(f, axis=1).max()) > force_cap:
                    continue
            valid.append((idx, at))

        ranked = list(reversed(valid))
        if offset > 0 and len(ranked) > offset:
            ranked = ranked[offset:]

        picked = apply_spacing_to_ranked_candidates(
            ranked,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, stype) for idx, at in picked]

    elif stype == "random":
        shuffled = list(candidates)
        random.shuffle(shuffled)

        picked = apply_spacing_to_ranked_candidates(
            shuffled,
            n,
            spacing,
            existing_selected_indices,
        )

        return [(idx, at, stype) for idx, at in picked]

    return []


def select_frames(run_dir, cfg):
    traj_file = run_dir / "traj.xyz"

    if not traj_file.exists():
        return [], []

    try:
        frames = read(traj_file, index=":")
        if not isinstance(frames, list):
            frames = [frames]
    except Exception:
        return [], []

    if not frames:
        return [], []

    ukey = cfg.get("uncertainty_key", "uncertainty")
    spacing = int(cfg.get("spacing", 0))

    # User-requested behavior: no distance compute when no filters are configured.
    distance_compute_enabled = bool(cfg.get("filters", []))

    # 1. Base check: ensure frames actually have the AL key evaluated.
    base_valid = [(i, at) for i, at in enumerate(frames) if ukey in at.arrays]

    if not base_valid:
        return [], []

    # Cache expensive per-frame scores so multiple selectors can reuse them.
    score_cache = {"min_distance": {}}

    # 2. Filter rejection criteria.
    valid_candidates = apply_filters(
        base_valid,
        cfg.get("filters", []),
        min_dist_cache=score_cache["min_distance"],
    )

    if not valid_candidates:
        return [], []

    status = get_run_status(run_dir)

    if status == "completed":
        selectors = cfg.get("selectors_completed", [])
    elif status == "failed":
        selectors = cfg.get("selectors_failed", [])
    else:
        # Fallback for unknown
        selectors = cfg.get("selectors_completed", [])

    selected_dict = {}

    # duplicate_policy determines if candidates are removed from evaluation pool.
    policy = cfg.get("duplicate_policy", "pick_next")

    remaining_candidates = list(valid_candidates)

    for scfg in selectors:
        existing_selected_indices = list(selected_dict.keys())

        picked = execute_selector(
            scfg,
            remaining_candidates,
            score_cache=score_cache,
            distance_compute_enabled=distance_compute_enabled,
            spacing=spacing,
            existing_selected_indices=existing_selected_indices,
        )

        # Track selected frames.
        for idx, at, reason in picked:
            if idx not in selected_dict or policy == "overwrite":
                at_copy = at.copy()
                at_copy.info["selection_strategy"] = reason
                at_copy.info["src_run"] = run_dir.name
                at_copy.info["src_status"] = status
                at_copy.info["src_index"] = idx
                selected_dict[idx] = at_copy

        if policy == "pick_next":
            selected_indices = list(selected_dict.keys())

            if spacing > 0:
                remaining_candidates = [
                    (idx, at)
                    for idx, at in remaining_candidates
                    if idx not in selected_dict
                    and is_spaced_from_selected(idx, selected_indices, spacing)
                ]
            else:
                remaining_candidates = [
                    (idx, at)
                    for idx, at in remaining_candidates
                    if idx not in selected_dict
                ]

    return valid_candidates, list(selected_dict.values())


def main():
    parser = argparse.ArgumentParser(description="Active Learning Frame Selection Module.")
    parser.add_argument("--config", type=str, required=True, help="Path to the active learning YAML configuration.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    cfg = full_config.get("selection_recipe", full_config)

    extxyz_path = Path(cfg.get("output_extxyz", "selected_frames.extxyz"))

    # 1. Clean previous artifact if it exists to avoid appending endlessly.
    if extxyz_path.exists():
        extxyz_path.unlink()

    base_path = Path(cfg.get("runs_dir", "./active_learning_runs"))

    runs = sorted(
        [p for p in base_path.glob("run_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])
        if p.name.split("_")[-1].isdigit()
        else 999999,
    )

    total_saved = 0
    ukey = cfg.get("uncertainty_key", "uncertainty")

    for run in tqdm(runs, desc="Selecting Frames"):
        _, selected = select_frames(run, cfg)

        # Write selected frames.
        for at in selected:
            at.info["al_key"] = ukey
            at.info["al_strategy"] = at.info.get("selection_strategy", "unknown")

            # Determine score: usually the max uncertainty of the frame for this key.
            if ukey in at.arrays:
                at.info["al_score"] = float(np.max(at.arrays[ukey]))
            else:
                at.info["al_score"] = 0.0

            write(extxyz_path, at, format="extxyz", append=True)
            total_saved += 1

    print(f"Saved {total_saved} frames to {extxyz_path}")


if __name__ == "__main__":
    main()