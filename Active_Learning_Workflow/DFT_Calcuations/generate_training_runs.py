"""
# generate_training_runs.py

Academic Software License (ASL) Copyright (c) 2026 Chris Davies @ University of Oxford.

## Usage
```
python generate_training_runs.py --extxyz_file <path> --incar_file <path> [--output_dir <path>] [--structure_name <prefix>] [--electron_delta <float>] [--gamma_only] [--set_langevin] [--extra_files <file1> <file2> ...]
python generate_training_runs.py --extxyz_file selected_frames_nequip.extxyz --incar_file INCAR --output_dir runs --gamma_only
```

## Arguments
- `--extxyz_file` (required): Path to the extended XYZ file containing multiple structures.
- `--incar_file` (required): Path to the INCAR file to be used as a template.
- `--output_dir` (optional): Directory where the structure directories will be created. Default is the current directory.
- `--structure_name` (optional): Prefix for naming each structure directory. Default is "structure_".
- `--electron_delta` (optional): Change in number of electrons from neutral system. Can be positive or negative. Default is 0.0.
- `--gamma_only` (optional): If set, configures the calculation for Gamma-point only (generates KPOINTS file, removes KSPACING/KGAMMA/KPAR from INCAR, sets KPAR=1).
- `--set_langevin` (optional): If set, automatically calculates and injects LANGEVIN_GAMMA for MD into the INCAR based on the DEFAULT_LANGEVIN_GAMMA dictionary.
- `--extra_files` (optional): Space-separated list of additional files to copy into each directory.


## Functionality
1. Reads structures from the provided extended XYZ file.
2. For each structure:
   - Creates a new directory in the output directory.
   - Generates a POSCAR file from the structure.
   - Creates a POTCAR file based on the elements in the structure.
   - Copies and modifies the INCAR file, adjusting the LANGEVIN_GAMMA parameter and NELECT if needed.
3. Handles element-specific POTCAR selections and Langevin damping parameters.
4. Calculates total number of valence electrons and applies specified delta for NELECT.
5. Provides logging information for each generated set of VASP input files.
"""

import os
import argparse
import shutil
from typing import List, Dict, Tuple
from ase.io import read
from ase.build.tools import sort
from pymatgen.io.vasp.inputs import Poscar, Potcar
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor # yeah I hate ths interconversion but ... hey ... 

# Default path for POTCAR files
DEFAULT_POTCAR_FOLDER = "{HPC}/potpaw_PBE.54"

# Dictionary mapping elements to their POTCAR files
DEFAULT_POTCARS = {
    'H':'H', 'He':'He', 'Li':'Li_sv', 'Be':'Be', 'B':'B', 'C':'C', 'N':'N', 'O':'O', 'F':'F', 'Na':'Na_pv',
    'Al':'Al', 'Si':'Si', 'P':'P', 'S':'S', 'Cl':'Cl', 'K':'K_sv', 'Ca':'Ca_sv', 'Sc':'Sc_sv', 'Ti':'Ti_sv',
    'V':'V_sv', 'Cr':'Cr_pv', 'Mn':'Mn_pv', 'Fe':'Fe', 'Co':'Co', 'Ni':'Ni', 'Cu':'Cu', 'Ga':'Ga_d', 'Ge':'Ge_d',
    'Y':'Y_sv', 'Zr':'Zr_sv', 'Mo':'Mo_sv', 'Ta':'Ta_pv', 'Ag':'Ag', 'La':'La', 'W':'W_sv', 'Br':'Br', 'I':'I', 'Zn':'Zn'
}

# Default Langevin gamma values (ps^-1) for specific elements
# Fallback for unlisted elements is 10.0
DEFAULT_LANGEVIN_GAMMA = {
    'Li': 10.0,
    'P': 10.0,
    'S': 10.0,
    'Br': 10.0,
    'Cl': 10.0,
    'Zn': 10.0
}


class VaspNVTInputGenerator:
    def __init__(self, 
                 extxyz_file: str, 
                 incar_path: str, 
                 output_dir: str,
                 structure_name: str,
                 electron_delta: float = 0.0,
                 gamma_only: bool = False,
                 set_langevin: bool = False,
                 extra_files: List[str] = None):
        """
        Initialize the VASP NVT input generator.
        
        Args:
            extxyz_file: Path to the extended XYZ file containing structures
            incar_path: Path to the template INCAR file
            output_dir: Directory to create structure folders
            structure_name: Prefix for structure directory names
            electron_delta: Change in number of electrons from neutral system
            gamma_only: Whether to generate a gamma-point only calculation
            set_langevin: Whether to calculate and insert LANGEVIN_GAMMA
            extra_files: List of additional files to copy into each directory
        """
        self.extxyz_file = extxyz_file
        self.incar_path = incar_path
        self.output_dir = output_dir
        self.structure_name = structure_name + "{}"
        self.electron_delta = electron_delta
        self.gamma_only = gamma_only
        self.set_langevin = set_langevin
        self.extra_files = extra_files or []

    def read_poscar_elements(self, poscar_path: str) -> Tuple[Dict[str, int], List[str]]:
        """
        Read elements and their counts from a POSCAR file.
        
        Args:
            poscar_path: Path to the POSCAR file
            
        Returns:
            Tuple containing:
            - Dictionary mapping element symbols to their counts
            - List of elements in order of appearance (for preserving POSCAR order)
        """
        with open(poscar_path, 'r') as fin:
            # Skip first 5 lines (comment, scale factor, and lattice vectors)
            for _ in range(5):
                fin.readline()
            
            # Read elements and their counts
            elements = fin.readline().strip().split()
            counts = [int(x) for x in fin.readline().strip().split()]
            
            # Create element count dictionary while preserving order
            element_counts = {}
            for element, count in zip(elements, counts):
                element_counts[element] = count
                
        return element_counts, elements

    def generate_potcar(self, element_list: List[str], directory: str) -> None:
        """
        Generate POTCAR file for the structure.
        
        Args:
            element_list: List of elements in order
            directory: Directory to create POTCAR in
        """
        potcar_path = os.path.join(directory, "POTCAR")
        try:
            with open(potcar_path, 'w') as fout:
                for element in element_list:
                    potcar_file = os.path.join(DEFAULT_POTCAR_FOLDER, DEFAULT_POTCARS[element], "POTCAR")
                    try:
                        with open(potcar_file, 'r') as fin:
                            fout.write(fin.read())
                    except IOError as e:
                        print(f"Error reading POTCAR for element {element}: {e}")
                        raise
        except IOError as e:
            print(f"Error writing POTCAR file: {e}")
            raise
    
    def get_nelect_line(self, n_valence_electrons) -> str:
        """
        Generate NELECT line for INCAR.
        
        Args:
            n_valence_electrons: Total number of valence electrons
            
        Returns:
            NELECT parameter line for INCAR
        """
        return f"NELECT = {n_valence_electrons:.1f}\n"

    def modify_incar(self, poscar_path: str, directory: str) -> None:
        """
        Modify INCAR file with LANGEVIN_GAMMA and NELECT parameters.
        
        Args:
            poscar_path: Path to the POSCAR file
            directory: Directory containing INCAR file
        """
        incar_file = os.path.join(directory, "INCAR")
        with open(incar_file, 'r') as file:
            incar_lines = file.readlines()

        # Get element information
        element_counts, element_list = self.read_poscar_elements(poscar_path)
        
        # Read ZVALs from generated POTCAR
        zvals = []
        potcar_path = os.path.join(directory, "POTCAR")
        with open(potcar_path, "r") as f:
            for line in f:
                if "ZVAL" in line:
                    parts = line.split(";")
                    for p in parts:
                        if "ZVAL" in p:
                            zvals.append(float(p.split("=")[1].split()[0]))
                            
        # Calculate NELECT
        total_valence = sum(element_counts[e] * zval for e, zval in zip(element_list, zvals))
        nelect = total_valence + self.electron_delta
        
        # Modify existing lines or add new ones
        modified_lines = []
        
        for line in incar_lines:
            strip_line = line.strip()
            if self.gamma_only and any(k in line for k in ["KSPACING", "KGAMMA", "KPAR"]):
                continue
            if strip_line.startswith("NELECT"):
                continue
            if self.set_langevin and strip_line.startswith("LANGEVIN_GAMMA"):
                continue
            modified_lines.append(line)
            
        if self.gamma_only:
            modified_lines.append("KPAR = 1\n")
            
        modified_lines.append(self.get_nelect_line(nelect))
        
        # Set Langevin Gamma based on species only if requested
        if self.set_langevin:
            langevin_gammas = [str(DEFAULT_LANGEVIN_GAMMA.get(e, 10.0)) for e in element_list]
            langevin_str = " ".join(langevin_gammas)
            modified_lines.append(f"LANGEVIN_GAMMA = {langevin_str}\n")
        
        with open(incar_file, 'w') as file:
            file.writelines(modified_lines)

    def generate_kpoints(self, directory: str) -> None:
        """
        Generate a Gamma-point only KPOINTS file.
        """
        kpoints_path = os.path.join(directory, "KPOINTS")
        with open(kpoints_path, 'w') as f:
            f.write("Gamma-point only\n0\nGamma\n1 1 1\n0 0 0\n")

    def generate_files(self) -> None:
        """
        Generate all VASP input files for each structure in the XYZ file.
        """
        # Read all structures from XYZ file
        structures = read(self.extxyz_file, index=':')
        
        # Process each structure
        for i, ase_atoms in enumerate(structures):
            # Create directory for this structure
            directory = os.path.join(self.output_dir, self.structure_name.format(i))
            os.makedirs(directory, exist_ok=True)
            
            # Sort atoms for consistency
            ase_atoms.wrap()
            ase_atoms = sort(ase_atoms)

            # Convert ASE Atoms to pymatgen Structure
            structure = AseAtomsAdaptor.get_structure(ase_atoms)

            # Generate POSCAR
            poscar = Poscar(structure)
            poscar_path = os.path.join(directory, "POSCAR")
            poscar.write_file(poscar_path)
            
            # Read elements from generated POSCAR
            element_counts, element_list = self.read_poscar_elements(poscar_path)
            
            # Generate POTCAR
            self.generate_potcar(element_list, directory)
            
            # Generate KPOINTS if gamma_only
            if self.gamma_only:
                self.generate_kpoints(directory)
            
            # Copy and modify INCAR
            try:
                dest_path = os.path.join(directory, 'INCAR')
                shutil.copy(self.incar_path, dest_path)
                self.modify_incar(poscar_path, directory)
                
                # Copy extra files
                for extra_file in self.extra_files:
                    if os.path.exists(extra_file):
                        shutil.copy(extra_file, directory)
                    else:
                        print(f"Warning: Extra file {extra_file} not found. Skipping.")
            except IOError as e:
                print(f"Error copying files: {e}")
                raise

            print(f"Generated VASP input files for NVT run in {directory}")

def main():
    """
    Parse command line arguments and run the VASP input generator.
    """
    parser = argparse.ArgumentParser(description="Generate VASP input files for NVT runs from extxyz file.")
    parser.add_argument("--extxyz_file", type=str, required=True, help="Path to the extxyz file containing multiple structures.")
    parser.add_argument("--incar_file", type=str, required=True, help="Path to the INCAR file to be used as a template.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where the structure directories will be created. Default is the current directory.")
    parser.add_argument("--structure_name", type=str, default="structure_", help="String which this code will add +i to for each structure in the extxyz")
    parser.add_argument("--electron_delta", type=float, default=0.0, help="Change in number of electrons from neutral system. Can be positive or negative. Default is 0.0.")
    parser.add_argument("--gamma_only", action="store_true", help="Set up the calculation for Gamma-point only (generates KPOINTS file, removes KSPACING/KGAMMA/KPAR from INCAR, sets KPAR=1)")
    parser.add_argument("--set_langevin", action="store_true", help="Automatically calculate and inject LANGEVIN_GAMMA for MD into the INCAR based on the DEFAULT_LANGEVIN_GAMMA dictionary.")
    parser.add_argument("--extra_files", nargs='+', default=[], help="List of additional files to copy into each generated directory.")

    args = parser.parse_args()
    
    generator = VaspNVTInputGenerator(
        args.extxyz_file,
        args.incar_file,
        args.output_dir,
        args.structure_name,
        args.electron_delta,
        args.gamma_only,
        args.set_langevin,
        args.extra_files
    )
    generator.generate_files()

if __name__ == "__main__":
    main()