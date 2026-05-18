# Argyrodite Zn Paper
A repository for the analysis scripts and potentials for "Effect of Zinc Incorporation on the Structure, Conductivity and (Electro)chemical Stability of Li6−2xZnxPS5X (X = Br, Cl, I) Argyrodite Solid Electrolytes"

**License:** Academic Software License (ASL). Copyright (c) 2026 Chris Davies @ University of Oxford.

Pipeline for developing Machine Learning Interatomic Potentials (MLIPs) for Zn-substituted Argyrodite solid-state electrolytes.

## Workflow Overview

This repository implements an iterative Active Learning (AL) loop to rapidly develop robust, highly accurate ML potentials while minimizing computational cost:

1. **Model Training (`Allegro_Config/`):** Train a committee of NequIP/Allegro models using available training data.
2. **Exploration (`Active_Learning_MD/`):** Run Molecular Dynamics (MD) using the current ML predictions to simulate behavior and explore new atomic configurations.
3. **Uncertainty Evaluation:** Use the model committee's variance across MD trajectories to detect out-of-domain structures.
4. **DFT Labeling (`DFT_Calcuations/`):** Extract the most uncertain or highly strained frames and calculate "ground truth" energies/forces using Density Functional Theory (VASP).
5. **Iteration (`Training_Data/`):** Append these newly labeled subsets into the training datasets and repeat the loop until the overall model accuracy is resilient. Final potential resides in `Models/`.

## Structure

- `Active_Learning_Workflow/Active_Learning_MD/`:
  - **`run_ase_md.py`:** Runs MD
  - **`run_committee.py`:** Evaluates MD trajectories with a model ensemble to compute uncertainty.
  - **`select_frames.py`:** Tools to filter and select the most informative frames (max stress, highest force, collisions) based on `al_config.yaml`.
  - **`analyse_al_dataset.py`:** Analyses the selected frames to understand the distribution of forces, stresses in the training dataset.

- `Active_Learning_Workflow/DFT_Calcuations/`:
  - **`generate_training_runs.py`:** Prepares the VASP calculation directories and inputs for the selected frames.
  - **`run_vasp.sh` & `check_convergence.py`:** Runs VASP for the selected frames and checks convergence.
  - **`get_xyz_from_vasp.py`:** Extracts the calculated energies, forces, and stresses back into `.extxyz` format for training.
  - **`INCAR`:** The standard DFT configuration template used for these static calculations.

- `Allegro_Config/`: Scripts and YAML configurations for training the NequIP/Allegro potential committee.

- `Training_Data/`: [TODO] The cumulative training datasets in `.extxyz` format.

- `Models/`: A model zip file with instructions on how to compile for deployment.