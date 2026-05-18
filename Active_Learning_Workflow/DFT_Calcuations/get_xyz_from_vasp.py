#!/usr/bin/env python3
"""
# get_xyz_from_vasp.py

Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

## Usage
```
python get_xyz_from_vasp.py [--search_dir <path>] [--prefix <string>] [--output <path>]
```

## Arguments
- `--search_dir` (optional): Directory containing the output VASP runs. Default is "runs".
- `--prefix` (optional): Prefix for naming each structure directory. Default is "structure_".
- `--output` (optional): Output filename for combined extxyz. Default is "combined_VASP.extxyz".

## Functionality
1. Scans the search directory for VASP OUTCAR files matching the structure prefix pattern.
2. Extracts atomic structures from completed runs sequentially.
3. Combines all read structures into a single extended XYZ file.
"""

import sys
import glob
import os
import argparse
from typing import List
from ase.io import read, write

def extract_xyz(search_dir: str, prefix: str, output: str) -> None:
    """
    Finds VASP OUTCAR files and combines them into an extxyz file.

    Args:
        search_dir: Directory containing the VASP run folders.
        prefix: Prefix for the structure folders (e.g. 'structure_').
        output: Extxyz filename to write combined structures into.
    """
    pattern = os.path.join(search_dir, f"{prefix}*/OUTCAR")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('/')[0]))

    atoms = []
    for f in files:
        try:
            atoms += read(f, index=":")
            print(f"Read {f}")
        except Exception as e:
            print(f"Skipping {f}: {e}")

    write(output, atoms, format="extxyz")
    print(f"\\nCombined {len(atoms)} frames into {output}")

def main() -> None:
    """
    Parse command line arguments and execute XYZ extraction.
    """
    parser = argparse.ArgumentParser(description="Extract and combine VASP OUTCARs into a single extxyz file.")
    parser.add_argument("--search_dir", type=str, default="runs", help="Directory where the structure directories are located.")
    parser.add_argument("--prefix", type=str, default="structure_", help="String prefix for structure directories.")
    parser.add_argument("--output", type=str, default="combined_VASP.extxyz", help="Output extxyz file name.")
    args = parser.parse_args()

    extract_xyz(args.search_dir, args.prefix, args.output)

if __name__ == "__main__":
    main()

