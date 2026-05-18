#!/usr/bin/env python3
"""
# check_convergence.py

Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

## Usage
```
python check_convergence.py [--root_dir <path>] [--max_files <int>] [--print_json] [--exit_on_fail] [--keep_failed]
```

## Arguments
- `--root_dir` (optional): Root directory to scan for VASP calculations. Default is "runs".
- `--max_files` (optional): Limit the number of directories to process.
- `--print_json` (optional): Print machine-readable JSON summary at the end.
- `--exit_on_fail` (optional): Exit with code 2 when any bad steps are found.
- `--keep_failed` (optional): Do not rename OUTCAR to OUTCAR_FAILED for non-converged jobs.

## Functionality
1. Scans recursively for INCAR files to identify VASP directories.
2. Extracts NELM from INCAR for each directory.
3. Checks OUTCAR tail to see if the run completed naturally.
4. Parses OSZICAR to find the number of ionic/electronic steps and whether NELM was hit.
"""

from pathlib import Path
import sys
import json
import re
import argparse

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def natural_sort_key(path: Path) -> list:
    """
    Sorts paths logically containing numbers (e.g., structure_2 before structure_10).

    Args:
        path: Path object to be sorted.

    Returns:
        List containing split string and integers for natural sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(path))]

def get_nelm(dirpath: Path) -> int:
    """
    Extract NELM from INCAR using regex parsing.

    Args:
        dirpath: Path to the directory containing the INCAR file.

    Returns:
        The extracted NELM integer, defaults to 60 if not found.
    """
    incar_file = dirpath / "INCAR"
    if incar_file.exists():
        try:
            with open(incar_file, "r") as f:
                for line in f:
                    # Ignore comments that might contain NELM
                    clean_line = line.split("#")[0].split("!")[0]
                    if "NELM" in clean_line:
                        match = re.search(r"NELM\s*=\s*(\d+)", clean_line)
                        if match:
                            return int(match.group(1))
        except Exception:
            pass
    return 60

def check_run_status(dirpath: Path) -> str:
    """
    Check OUTCAR tail to see if the run finished naturally.

    Args:
        dirpath: Path to the directory containing the OUTCAR file.

    Returns:
        Status string ("COMPLETED", "RUNNING", "PENDING", or "ERROR").
    """
    outcar_file = dirpath / "OUTCAR"
    if not outcar_file.exists():
        outcar_failed = dirpath / "OUTCAR_FAILED"
        if outcar_failed.exists():
            outcar_file = outcar_failed
        else:
            return "PENDING"
    try:
        with open(outcar_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - 2000, 0))
            tail = f.read().decode("utf-8", errors="ignore")
            if "General timing and accounting" in tail or "Voluntary context switches" in tail:
                return "COMPLETED"
            return "RUNNING"
    except Exception:
        return "ERROR"

def check_oszicar_convergence(dirpath: Path, nelm: int) -> dict:
    """
    Parse OSZICAR (or vasp.out) to track ionic steps, electronic steps, and NELM hits.

    Args:
        dirpath: Path to the directory containing OSZICAR or vasp.out.
        nelm: Maximum number of electron steps allowed (from INCAR).

    Returns:
        Dictionary containing tracked convergence metrics.
    """
    oszicar = dirpath / "OSZICAR"
    vasp_out = dirpath / "vasp.out"
    
    # Prefer vasp.out if it exists as it might be updated more frequently than OSZICAR
    target_file = vasp_out if vasp_out.exists() else oszicar

    result = {"ionic_steps": 0, "bad_steps": 0, "hit_nelm": False, "last_scf": 0}
    
    if not target_file.exists():
        return result

    # Match electronic steps (e.g., "DAV: 150", "RMM:  15", "ALGO:  3")
    el_pattern = re.compile(r"^\s*(?:DAV|RMM|CG\w*|SDA|CGA|ALGO)\s*:\s*(\d+)")
    # Match ionic step completions (e.g., "   1 F= -.2325...")
    ion_pattern = re.compile(r"^\s*(\d+)\s+F=")

    bad_in_current_ion = False

    try:
        with open(target_file, "r") as f:
            for line in f:
                el_match = el_pattern.match(line)
                if el_match:
                    el_step = int(el_match.group(1))
                    result["last_scf"] = el_step  # Constantly update with the latest SCF step
                    
                    if el_step >= nelm and not bad_in_current_ion:
                        result["bad_steps"] += 1
                        result["hit_nelm"] = True
                        bad_in_current_ion = True
                else:
                    ion_match = ion_pattern.match(line)
                    if ion_match:
                        result["ionic_steps"] += 1
                        bad_in_current_ion = False  # Reset for the next ionic step
    except Exception:
        pass

    return result

def truncate_path(p: str, max_len=45) -> str:
    """
    Truncate file path string to keep table rows clean and aligned.

    Args:
        p: String representation of the path.
        max_len: Maximum allowed length before truncating.

    Returns:
        Truncated path string.
    """
    return "..." + p[-(max_len-3):] if len(p) > max_len else p

def main() -> None:
    """
    Parse command line arguments and execute convergence check.
    """
    parser = argparse.ArgumentParser(description="VASP OUTCAR/OSZICAR convergence checker.")
    parser.add_argument("--root_dir", type=str, default="runs", help="Root directory to scan for VASP calculations.")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of directories to process.")
    parser.add_argument("--print_json", action="store_true", help="Print machine-readable JSON summary at end.")
    parser.add_argument("--exit_on_fail", action="store_true", help="Exit with code 2 when any bad steps are found.")
    parser.add_argument("--keep_failed", action="store_true", help="Do not rename OUTCAR to OUTCAR_FAILED for non-converged jobs. Defaults to False (meaning renaming happens by default).")
    args = parser.parse_args()

    root_path = Path(args.root_dir)

    print(f"Scanning for VASP directories in {root_path}...", end="", flush=True)
    directories = []
    for p in root_path.rglob("INCAR"):
        directories.append(p.parent)
        if len(directories) % 50 == 0:
            print(f"\rScanning for VASP directories in {root_path}... Found {len(directories)}", end="", flush=True)
    print(f"\rScanning for VASP directories in {root_path}... Found {len(directories)} total.")

    if not directories:
        print(f"{Colors.YELLOW}No calculations found to process.{Colors.RESET}")
        return

    directories = sorted(directories, key=natural_sort_key)

    if args.max_files:
        directories = directories[:args.max_files]

    results = []
    overall_bad = 0
    completed_count = 0
    pending_count = 0

    total_dirs = len(directories)
    for i, d in enumerate(directories):
        print(f"\rProcessing directory {i+1}/{total_dirs}... ({d.name})", end="", flush=True)
        status = check_run_status(d)
        nelm = get_nelm(d)
        conv = check_oszicar_convergence(d, nelm)
        
        if status == "COMPLETED":
            completed_count += 1
        elif status == "PENDING":
            pending_count += 1

        results.append({
            "path": str(d),
            "status": status,
            "nelm": nelm,
            "ionic_steps": conv["ionic_steps"],
            "last_scf": conv["last_scf"],
            "bad_steps": conv["bad_steps"],
            "converged": not conv["hit_nelm"]
        })
        overall_bad += conv["bad_steps"]
        
        # Rename OUTCAR if completed but not converged
        if not args.keep_failed and status == "COMPLETED" and not results[-1]["converged"]:
            outcar_path = d / "OUTCAR"
            if outcar_path.exists():
                try:
                    outcar_path.rename(d / "OUTCAR_FAILED")
                except Exception as e:
                    pass

    print("\r" + " " * 80 + "\r", end="", flush=True)  # Clear the reading progress line

    print(f"\n{Colors.BOLD}{'='*95}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'VASP CONVERGENCE REPORT':^95}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*95}{Colors.RESET}")
    
    header = f"{'DIRECTORY':<45} | {'STATUS':<9} | {'NELM':<4} | {'IONIC':<5} | {'SCF':<4} | {'BAD':<3} | {'CONV':<4}"
    print(header)
    print("-" * 95)

    for r in results:
        p_trunc = truncate_path(r["path"], 45)
        
        # Colorize Status
        if r["status"] == "COMPLETED":
            if r["converged"]:
                stat_str = f"{Colors.GREEN}{r['status']:<9}{Colors.RESET}"
            else:
                stat_str = f"{Colors.RED}{'FAILED':<9}{Colors.RESET}"
        elif r["status"] == "RUNNING":
            stat_str = f"{Colors.YELLOW}{r['status']:<9}{Colors.RESET}"
        elif r["status"] == "PENDING":
            stat_str = f"{Colors.CYAN}{r['status']:<9}{Colors.RESET}"
        else:
            stat_str = f"{Colors.RED}{r['status']:<9}{Colors.RESET}"
        
        # Determine convergence icon
        if r["status"] == "PENDING":
            icon = f"{Colors.CYAN}[-]{Colors.RESET}"
        elif r["status"] == "MISSING":
            icon = f"{Colors.RED}[?]{Colors.RESET}"
        elif r["status"] == "RUNNING":
            icon = f"{Colors.YELLOW}[⋯]{Colors.RESET}" if r["converged"] else f"{Colors.RED}no {Colors.RESET}"
        else:
            icon = f"{Colors.GREEN}yes{Colors.RESET}" if r["converged"] else f"{Colors.RED}no {Colors.RESET}"

        # Highlight Bad steps if > 0
        bad_str = f"{Colors.RED}{r['bad_steps']:<3}{Colors.RESET}" if r['bad_steps'] > 0 else f"{r['bad_steps']:<3}"

        print(f"{p_trunc:<45} | {stat_str} | {r['nelm']:<4} | {r['ionic_steps']:<5} | {r['last_scf']:<4} | {bad_str} | {icon:<4}")

    print("-" * 95)
    
    total = len(results)
    started = total - pending_count

    failed_count = sum(1 for r in results if not r['converged'] and r['status'] != 'PENDING')
    success_count = sum(1 for r in results if r['converged'] and r['status'] == 'COMPLETED')
    
    bar_len = 40
    
    # Overall Progress
    pct_overall = int((completed_count / total) * 100) if total > 0 else 0
    f_len = int((sum(1 for r in results if not r['converged'] and r['status'] == 'COMPLETED') / total) * bar_len) if total > 0 else 0
    s_len = int((success_count / total) * bar_len) if total > 0 else 0
    
    # Fix rounding visually
    filled_overall = int((completed_count / total) * bar_len) if total > 0 else 0
    if s_len + f_len < filled_overall:
        s_len += filled_overall - (s_len + f_len)
        
    bar_overall = f"{Colors.GREEN}{'█' * s_len}{Colors.RED}{'█' * f_len}{Colors.RESET}{'-' * (bar_len - filled_overall)}"

    # Started Runs Progress
    pct_started = int((completed_count / started) * 100) if started > 0 else 0
    f_len_s = int((sum(1 for r in results if not r['converged'] and r['status'] == 'COMPLETED') / started) * bar_len) if started > 0 else 0
    s_len_s = int((success_count / started) * bar_len) if started > 0 else 0
    
    filled_started = int((completed_count / started) * bar_len) if started > 0 else 0
    if s_len_s + f_len_s < filled_started:
        s_len_s += filled_started - (s_len_s + f_len_s)

    bar_started = f"{Colors.GREEN}{'█' * s_len_s}{Colors.RED}{'█' * f_len_s}{Colors.RESET}{'-' * (bar_len - filled_started)}"
    
    print(f"{Colors.BOLD}OVERALL PROGRESS: [{bar_overall}{Colors.BOLD}] {pct_overall}%{Colors.RESET} {Colors.BOLD}({completed_count}/{total} Total Completed) [{failed_count} failed]{Colors.RESET}")
    print(f"{Colors.BOLD}STARTED PROGRESS: [{bar_started}{Colors.BOLD}] {pct_started}%{Colors.RESET} {Colors.BOLD}({completed_count}/{started} Started Completed) [{failed_count} failed]{Colors.RESET}")
    print(f"TOTAL PENDING: {pending_count} calculations")
    print(f"TOTAL FAILED : {failed_count} calculations (Total {overall_bad} bad ionic steps)")
    print(f"{Colors.BOLD}{'='*95}{Colors.RESET}\n")

    if args.print_json:
        print("JSON SUMMARY:")
        print(json.dumps({"total_checked": len(results), "calculations": results}, indent=2))

    if args.exit_on_fail and overall_bad > 0:
        sys.exit(2)

if __name__ == "__main__":
    main()