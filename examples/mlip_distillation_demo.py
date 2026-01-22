#!/usr/bin/env python3
"""
MLIP Distillation Demo Script

This script demonstrates the workflow for distilling Machine Learning Interatomic Potentials (ORB)
into classical GULP potentials using GGen.

Workflow:
1. Collect diverse training data using ORB (collect_comprehensive)
2. Filter similar structures for dataset quality (DuplicateFilter)
3. Fit GULP potentials using QEq model and native fitting
4. Validate the fitted potential
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.db import connect
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure ggen can be imported if running from source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ggen import (
    get_orb_calculator, 
    MLIPCollector, 
    CollectionConfig,
    DuplicateFilter, 
    filter_training_data,
    FitTarget,
    PotentialConfig, 
    ChargeModel, 
    PotentialType,
    BuckinghamParams, 
    build_qeq_config, 
    QEQ_PARAMS_DATABASE,
    GULPFitter,
    get_gulp_calculator
)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Set GULP paths (modify for your system)
# Warning: You must ensure these point to your valid GULP executable and library path
if "GULP_EXE" not in os.environ:
    os.environ["GULP_EXE"] = "/path/to/gulp"  # CHANGE THIS
if "GULP_LIB" not in os.environ:
    os.environ["GULP_LIB"] = "/path/to/gulp/Libraries"  # CHANGE THIS

print("--- MLIP Distillation Demo ---")

# -----------------------------------------------------------------------------
# Step 1: Comprehensive Data Collection
# -----------------------------------------------------------------------------
print("\nStep 1: Comprehensive Data Collection")

# Create ORB calculator
# Note: Requires GPU for efficiency, set device='cpu' if needed
try:
    calc = get_orb_calculator(device="cuda")
except Exception as e:
    print(f"Warning: Could not load ORB calculator on CUDA, trying CPU. Error: {e}")
    calc = get_orb_calculator(device="cpu")

# Configure collection
config = CollectionConfig(
    db_path="comprehensive_training.db",
    md_temperatures=[300, 600, 1000],
    md_steps_per_temp=50,
    md_save_interval=5,
    rattle_stdev=[0.01, 0.05],
    rattle_n_configs=5,
)

collector = MLIPCollector(calculator=calc, config=config)

# Load CIF files
# IMPORTANT: This assumes you have a 'cif_files' directory. 
# For demo purposes, we'll check if it exists or use dummy data if possible (not implemented here).
cif_folder = Path("./cif_files")
cif_files = list(cif_folder.glob("*.cif"))

if not cif_files:
    print(f"Warning: No CIF files found in {cif_folder}. Skipping actual collection loop.")
    print("Please place .cif files in './cif_files' to run data collection.")
else:
    for cif_file in cif_files[:3]:
        atoms = read(cif_file)
        name = cif_file.stem
        
        print(f"Collecting data for {name}...")
        # Comprehensive collection (all methods at once!)
        collector.collect_comprehensive(
            atoms,
            name=name,
            include_original=True,   # MD on original
            include_scaled=True,     # Volume scaling + MD
            include_sheared=True,    # Shear strain + MD
            include_rattling=True,   # Random displacements
            include_strain=True,     # Strain perturbations
            scale_factors=[0.90, 0.95, 1.05, 1.10],
        )

print(collector.stats.summary())

# -----------------------------------------------------------------------------
# Step 2: Filter Similar Structures
# -----------------------------------------------------------------------------
print("\nStep 2: Filter Similar Structures")

# Using the convenience function
# Note: Ensure dscribe is installed for 'soap' method
if os.path.exists("comprehensive_training.db"):
    result = filter_training_data(
        "comprehensive_training.db",
        "filtered_training.db",
        method="soap",      # or "coulomb"
        threshold=0.95,     # similarity threshold
    )
    print(result.summary())
    
    # Visualization (Similarity Matrix)
    print("Generating similarity matrix plot...")
    filter_obj = DuplicateFilter(method="soap", threshold=0.90)
    db = connect("filtered_training.db")
    atoms_list = [row.toatoms() for row in list(db.select())[:20]]
    
    if atoms_list:
        sim_matrix = filter_obj.get_similarity_matrix(atoms_list)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(sim_matrix, cmap='viridis')
        plt.colorbar(label='Similarity')
        plt.title('Structure Similarity Matrix')
        plt.savefig('structure_similarity.png')
        print("Saved structure_similarity.png")
        plt.close()
else:
    print("Skipping filtering (database not found).")
    # Setup dummy db for testing if needed, or just exit/skip

# -----------------------------------------------------------------------------
# Step 3: Prepare Fitting Targets
# -----------------------------------------------------------------------------
print("\nStep 3: Prepare Fitting Targets")

targets = []
if os.path.exists("filtered_training.db"):
    db = connect("filtered_training.db")
    MAX_FORCE = 50.0

    for row in db.select():
        if row.get("max_force", 0) < MAX_FORCE:
            atoms = row.toatoms()
            targets.append(FitTarget(
                name=f"s_{row.id}",
                atoms=atoms,
                energy=row.get("total_energy"),
                energy_weight=1.0,
            ))
    print(f"Prepared {len(targets)} training targets")
else:
    print("No filtered database found. Cannot prepare targets.")

# -----------------------------------------------------------------------------
# Step 4: Configure Potential (QEq with Auto-Loaded Parameters)
# -----------------------------------------------------------------------------
print("\nStep 4: Configure Potential")

# Auto-build QEq config
qeq_params = build_qeq_config(["Nb", "W", "O"])

config_qeq = PotentialConfig(
    name="NbWO_qeq",
    charge_model=ChargeModel.QEQ,
    potential_type=PotentialType.BUCKINGHAM,
    buckingham={
        ("Nb", "O"): BuckinghamParams(A=5000, rho=0.35, C=0),
        ("W", "O"): BuckinghamParams(A=6000, rho=0.33, C=0),
        ("O", "O"): BuckinghamParams(A=22000, rho=0.15, C=28),
    },
    qeq_params=qeq_params,
    fit_buckingham=True,
    fit_charges=True,
)
print("Potential configuration created.")

# -----------------------------------------------------------------------------
# Step 5: Fit Potential (Native GULP Fitting)
# -----------------------------------------------------------------------------
print("\nStep 5: Fit Potential (Native)")

if targets and os.path.exists(os.environ.get("GULP_EXE", "")):
    fitter = GULPFitter(
        gulp_command=os.environ["GULP_EXE"],
        gulp_lib=os.environ["GULP_LIB"],
        verbose=True,
    )

    # NATIVE fitting (fast! - single GULP call)
    result = fitter.fit_native(
        config=config_qeq,
        targets=targets,
        fit_cycles=100,        # GULP internal cycles
        simultaneous=True,     # Fit all structures at once
        relax_structures=True, # Optimize during fitting
    )

    print(f"\nFitting {'converged' if result.converged else 'failed'}")
    print(f"Objective: {result.objective_value:.4e}")

    # Save fitted potential
    fitter.save_potential(result.config, "NbWO_fitted.lib")

    print("\nFitted Buckingham parameters:")
    for pair, params in result.config.buckingham.items():
        print(f"  {pair}: A={params.A:.1f}, rho={params.rho:.4f}, C={params.C:.1f}")

    # -----------------------------------------------------------------------------
    # Step 6: Validation
    # -----------------------------------------------------------------------------
    print("\nStep 6: Validation")

    # Create calculator with fitted potential
    gulp_calc = get_gulp_calculator(
        library="NbWO_fitted.lib",
        keywords="conp gradients qeq",
    )

    orb_e, gulp_e = [], []

    print("Calculating energies for validation set (first 30 targets)...")
    for target in targets[:30]:
        atoms = target.atoms.copy()
        n = len(atoms)
        orb_e.append(target.energy / n)
        
        atoms.calc = gulp_calc
        try:
            gulp_e.append(atoms.get_potential_energy() / n)
        except Exception:
            gulp_e.append(np.nan)

    valid = ~np.isnan(gulp_e)
    if any(valid):
        mae = mean_absolute_error(np.array(orb_e)[valid], np.array(gulp_e)[valid])
        r2 = r2_score(np.array(orb_e)[valid], np.array(gulp_e)[valid])

        print(f"MAE: {mae:.4f} eV/atom")
        print(f"R²: {r2:.4f}")

        # Parity plot
        print("Generating parity plot...")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(np.array(orb_e)[valid], np.array(gulp_e)[valid], alpha=0.6)
        
        # Plot y=x line
        min_val = min(min(np.array(orb_e)[valid]), min(np.array(gulp_e)[valid]))
        max_val = max(max(np.array(orb_e)[valid]), max(np.array(gulp_e)[valid]))
        lims = [min_val, max_val]
        
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('ORB Energy (eV/atom)')
        ax.set_ylabel('GULP Energy (eV/atom)')
        ax.set_title(f'Energy Parity (MAE={mae:.4f}, R²={r2:.4f})')
        plt.tight_layout()
        plt.savefig('energy_parity.png')
        print("Saved energy_parity.png")
        plt.close()
    else:
        print("No valid GULP energies calculated.")

else:
    print("Skipping fitting: targets not ready or GULP_EXE not found/set.")
    print("Make sure 'comprehensive_training.db' exists and 'GULP_EXE' path is correct.")

print("\nDemo script finished.")
