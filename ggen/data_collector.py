"""
Data collector for GULP potential fitting via MLIP distillation.

Generates training data by:
1. MD sampling at various temperatures
2. Phonon displacement sampling
3. Composition substitution
4. Volume/strain perturbations
5. Geometry optimization for equilibrium structures
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import warnings

from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS, FIRE
from ase.constraints import ExpCellFilter, StrainFilter
from ase.calculators.calculator import Calculator
from ase import units


@dataclass
class TrainingData:
    """Single training data point."""
    structure: Atoms
    energy: float  # eV
    forces: np.ndarray  # eV/Å, shape (n_atoms, 3)
    stress: Optional[np.ndarray] = None  # eV/Å³, Voigt notation
    source: str = ""  # Origin of this structure
    metadata: Optional[Dict] = None


class DataCollector:
    """
    Collect training data for GULP potential fitting.

    Uses an MLIP calculator as teacher to generate energy/force labels.

    Args:
        calculator: ASE calculator (MLIP) for energy/force evaluation
        n_workers: Parallel workers for batch calculations

    Example:
        from orb_models.forcefield import pretrained
        calc = pretrained.orb_v2()

        collector = DataCollector(calc)

        # Collect from MD
        data = collector.sample_md(atoms, temperatures=[300, 600, 1000])

        # Collect from phonon sampling
        data += collector.sample_phonon_displacements(atoms)

        # Save for GULP fitting
        collector.save_gulp_format(data, "training_data.gin")
    """

    def __init__(
        self,
        calculator: Calculator,
        n_workers: int = 1,
    ):
        self.calculator = calculator
        self.n_workers = n_workers
        self._data: List[TrainingData] = []

    def _calculate_single(self, atoms: Atoms, source: str = "") -> TrainingData:
        """Calculate energy and forces for single structure."""
        atoms = atoms.copy()
        atoms.calc = self.calculator

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        stress = None
        try:
            stress = atoms.get_stress()
        except:
            pass

        return TrainingData(
            structure=atoms,
            energy=energy,
            forces=forces.copy(),
            stress=stress,
            source=source,
        )

    def calculate_batch(
        self,
        atoms_list: List[Atoms],
        source: str = "",
    ) -> List[TrainingData]:
        """Calculate energy/forces for multiple structures."""
        results = []
        for i, atoms in enumerate(atoms_list):
            try:
                data = self._calculate_single(atoms, f"{source}_{i}")
                results.append(data)
            except Exception as e:
                warnings.warn(f"Calculation failed for structure {i}: {e}")
        return results

    # ==================== Equilibrium Structures ====================

    def optimize_structure(
        self,
        atoms: Atoms,
        fmax: float = 0.01,
        steps: int = 500,
        optimize_cell: bool = True,
    ) -> TrainingData:
        """
        Optimize structure to get equilibrium configuration.

        Args:
            atoms: Initial structure
            fmax: Force convergence criterion (eV/Å)
            steps: Maximum optimization steps
            optimize_cell: Whether to optimize cell parameters

        Returns:
            TrainingData for optimized structure
        """
        atoms = atoms.copy()
        atoms.calc = self.calculator

        if optimize_cell:
            filtered = ExpCellFilter(atoms)
            opt = LBFGS(filtered, logfile=None)
        else:
            opt = LBFGS(atoms, logfile=None)

        opt.run(fmax=fmax, steps=steps)

        return self._calculate_single(atoms, "optimized")

    def collect_equilibrium_structures(
        self,
        atoms_list: List[Atoms],
        fmax: float = 0.01,
    ) -> List[TrainingData]:
        """Optimize multiple structures to equilibrium."""
        results = []
        for i, atoms in enumerate(atoms_list):
            try:
                data = self.optimize_structure(atoms, fmax=fmax)
                data.source = f"equilibrium_{i}"
                results.append(data)
            except Exception as e:
                warnings.warn(f"Optimization failed for structure {i}: {e}")
        return results

    # ==================== MD Sampling ====================

    def sample_md(
        self,
        atoms: Atoms,
        temperatures: List[float] = [300, 600, 1000],
        n_steps: int = 1000,
        sample_interval: int = 50,
        timestep: float = 1.0,  # fs
        equilibration_steps: int = 100,
    ) -> List[TrainingData]:
        """
        Sample structures from MD trajectories at various temperatures.

        Args:
            atoms: Initial structure (should be pre-optimized)
            temperatures: List of temperatures in K
            n_steps: MD steps per temperature
            sample_interval: Steps between samples
            timestep: MD timestep in fs
            equilibration_steps: Steps to equilibrate before sampling

        Returns:
            List of TrainingData from MD snapshots
        """
        results = []

        for T in temperatures:
            atoms_md = atoms.copy()
            atoms_md.calc = self.calculator

            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms_md, temperature_K=T)

            # Setup Langevin dynamics
            dyn = Langevin(
                atoms_md,
                timestep=timestep * units.fs,
                temperature_K=T,
                friction=0.01,
                logfile=None,
            )

            # Equilibration
            dyn.run(equilibration_steps)

            # Production run with sampling
            for step in range(n_steps):
                dyn.run(1)

                if step % sample_interval == 0:
                    try:
                        data = self._calculate_single(
                            atoms_md,
                            f"md_T{T}K_step{step}"
                        )
                        results.append(data)
                    except:
                        pass

        return results

    def sample_md_diverse(
        self,
        atoms: Atoms,
        temperature_range: Tuple[float, float] = (100, 1500),
        n_temperature_points: int = 10,
        n_trajectories_per_temp: int = 2,
        n_steps: int = 500,
        sample_interval: int = 25,
        timestep: float = 1.0,
        friction_values: List[float] = [0.002, 0.01, 0.05],
        heating_cooling_ramps: bool = True,
        ramp_steps: int = 200,
    ) -> List[TrainingData]:
        """
        Diverse MD sampling with multiple temperatures, trajectories, and ensembles.

        This method provides more comprehensive configuration space coverage by:
        1. Spanning a wide temperature range
        2. Running multiple independent trajectories at each temperature
        3. Varying friction coefficients (effective thermostat coupling)
        4. Including heating/cooling ramps to sample transition states

        Args:
            atoms: Initial structure (should be pre-optimized)
            temperature_range: (T_min, T_max) in Kelvin
            n_temperature_points: Number of temperature points in range
            n_trajectories_per_temp: Independent trajectories per temperature
            n_steps: Production MD steps per trajectory
            sample_interval: Steps between samples
            timestep: MD timestep in fs
            friction_values: List of friction coefficients to sample
            heating_cooling_ramps: Whether to include heating/cooling ramps
            ramp_steps: Steps for temperature ramps

        Returns:
            List of TrainingData from diverse MD sampling
        """
        results = []
        T_min, T_max = temperature_range
        temperatures = np.linspace(T_min, T_max, n_temperature_points)

        print(f"  Diverse MD: {n_temperature_points} temps × "
              f"{n_trajectories_per_temp} traj × {len(friction_values)} frictions")

        # Standard NVT sampling at each temperature
        for T in temperatures:
            for traj_idx in range(n_trajectories_per_temp):
                # Vary friction for different sampling behavior
                friction = friction_values[traj_idx % len(friction_values)]

                atoms_md = atoms.copy()
                atoms_md.calc = self.calculator

                # Random initial velocities for each trajectory
                MaxwellBoltzmannDistribution(atoms_md, temperature_K=T)

                dyn = Langevin(
                    atoms_md,
                    timestep=timestep * units.fs,
                    temperature_K=T,
                    friction=friction,
                    logfile=None,
                )

                # Equilibration
                dyn.run(50)

                # Production
                for step in range(n_steps):
                    dyn.run(1)
                    if step % sample_interval == 0:
                        try:
                            data = self._calculate_single(
                                atoms_md,
                                f"md_T{T:.0f}K_traj{traj_idx}_f{friction}_step{step}"
                            )
                            results.append(data)
                        except:
                            pass

        # Heating/cooling ramps for transition sampling
        if heating_cooling_ramps and len(temperatures) >= 2:
            print(f"  Adding heating/cooling ramps...")
            # Heating ramp
            atoms_ramp = atoms.copy()
            atoms_ramp.calc = self.calculator
            MaxwellBoltzmannDistribution(atoms_ramp, temperature_K=T_min)

            for step in range(ramp_steps):
                # Linearly increase temperature
                T_current = T_min + (T_max - T_min) * step / ramp_steps
                dyn = Langevin(
                    atoms_ramp,
                    timestep=timestep * units.fs,
                    temperature_K=T_current,
                    friction=0.01,
                    logfile=None,
                )
                dyn.run(5)

                if step % (sample_interval // 2) == 0:
                    try:
                        data = self._calculate_single(
                            atoms_ramp,
                            f"md_heating_T{T_current:.0f}K_step{step}"
                        )
                        results.append(data)
                    except:
                        pass

            # Cooling ramp
            MaxwellBoltzmannDistribution(atoms_ramp, temperature_K=T_max)
            for step in range(ramp_steps):
                T_current = T_max - (T_max - T_min) * step / ramp_steps
                dyn = Langevin(
                    atoms_ramp,
                    timestep=timestep * units.fs,
                    temperature_K=T_current,
                    friction=0.01,
                    logfile=None,
                )
                dyn.run(5)

                if step % (sample_interval // 2) == 0:
                    try:
                        data = self._calculate_single(
                            atoms_ramp,
                            f"md_cooling_T{T_current:.0f}K_step{step}"
                        )
                        results.append(data)
                    except:
                        pass

        return results

    def sample_md_npt(
        self,
        atoms: Atoms,
        temperatures: List[float] = [300, 600, 1000],
        pressures: List[float] = [0.0, 0.1, 1.0],  # GPa
        n_steps: int = 500,
        sample_interval: int = 25,
        timestep: float = 1.0,
    ) -> List[TrainingData]:
        """
        NPT-like sampling by combining NVT MD with cell relaxation.

        GULP/classical potentials often need P-T data. This approximates
        NPT by alternating MD steps with cell optimization.

        Args:
            atoms: Initial structure
            temperatures: Temperatures in K
            pressures: Pressures in GPa
            n_steps: MD steps per (T, P) pair
            sample_interval: Steps between samples
            timestep: MD timestep in fs

        Returns:
            List of TrainingData
        """
        from ase.constraints import UnitCellFilter

        results = []

        for T in temperatures:
            for P in pressures:
                atoms_md = atoms.copy()
                atoms_md.calc = self.calculator

                # Apply external pressure via cell filter
                if P != 0.0:
                    # Scale cell to approximate pressure
                    # Using bulk modulus estimate ~100 GPa for oxides
                    B_estimate = 100.0  # GPa
                    strain = -P / (3 * B_estimate)
                    scale = (1 + strain) ** (1/3)
                    atoms_md.set_cell(atoms_md.cell * scale, scale_atoms=True)

                    # Relax cell at this pressure
                    try:
                        ucf = UnitCellFilter(atoms_md)
                        opt = LBFGS(ucf, logfile=None)
                        opt.run(fmax=0.1, steps=30)
                    except:
                        pass

                MaxwellBoltzmannDistribution(atoms_md, temperature_K=T)
                dyn = Langevin(
                    atoms_md,
                    timestep=timestep * units.fs,
                    temperature_K=T,
                    friction=0.01,
                    logfile=None,
                )

                dyn.run(50)  # equilibration

                for step in range(n_steps):
                    dyn.run(1)
                    if step % sample_interval == 0:
                        try:
                            data = self._calculate_single(
                                atoms_md,
                                f"md_T{T:.0f}K_P{P:.1f}GPa_step{step}"
                            )
                            results.append(data)
                        except:
                            pass

        return results

    # ==================== Phonon Sampling ====================

    def sample_phonon_displacements(
        self,
        atoms: Atoms,
        displacement: float = 0.01,  # Å
        n_displacements: int = 2,  # per atom per direction
    ) -> List[TrainingData]:
        """
        Generate structures with phonon-like displacements.

        Args:
            atoms: Equilibrium structure
            displacement: Displacement magnitude in Å
            n_displacements: Number of displacements per atom per direction

        Returns:
            List of TrainingData for displaced structures
        """
        results = []
        n_atoms = len(atoms)

        for i in range(n_atoms):
            for direction in range(3):  # x, y, z
                for sign in [-1, 1]:
                    for mag in np.linspace(0.5, 1.0, n_displacements):
                        displaced = atoms.copy()
                        d = displacement * mag * sign
                        displaced.positions[i, direction] += d

                        try:
                            data = self._calculate_single(
                                displaced,
                                f"phonon_atom{i}_dir{direction}_d{d:.3f}"
                            )
                            results.append(data)
                        except:
                            pass

        return results

    def sample_random_displacements(
        self,
        atoms: Atoms,
        n_samples: int = 100,
        max_displacement: float = 0.1,  # Å
        temperature_scale: bool = True,
    ) -> List[TrainingData]:
        """
        Generate structures with random atomic displacements.

        Args:
            atoms: Equilibrium structure
            n_samples: Number of samples to generate
            max_displacement: Maximum displacement per atom
            temperature_scale: Scale displacements by sqrt(T) distribution

        Returns:
            List of TrainingData
        """
        results = []

        for i in range(n_samples):
            displaced = atoms.copy()

            if temperature_scale:
                # Gaussian displacements (thermal-like)
                displacements = np.random.randn(len(atoms), 3) * max_displacement / 3
            else:
                # Uniform random
                displacements = (np.random.rand(len(atoms), 3) - 0.5) * 2 * max_displacement

            displaced.positions += displacements

            try:
                data = self._calculate_single(displaced, f"random_disp_{i}")
                results.append(data)
            except:
                pass

        return results

    # ==================== Volume/Strain Perturbations ====================

    def sample_volume_strain(
        self,
        atoms: Atoms,
        strain_range: Tuple[float, float] = (-0.05, 0.05),
        n_samples: int = 11,
    ) -> List[TrainingData]:
        """
        Generate structures with isotropic volume strains (E-V curve).

        Args:
            atoms: Equilibrium structure
            strain_range: (min, max) volumetric strain
            n_samples: Number of strain points

        Returns:
            List of TrainingData
        """
        results = []
        strains = np.linspace(strain_range[0], strain_range[1], n_samples)

        for strain in strains:
            strained = atoms.copy()
            # Isotropic scaling: V' = V(1+e), so linear scale = (1+e)^(1/3)
            scale = (1 + strain) ** (1/3)
            strained.set_cell(atoms.cell * scale, scale_atoms=True)

            try:
                data = self._calculate_single(strained, f"volume_strain_{strain:.3f}")
                results.append(data)
            except:
                pass

        return results

    def sample_shear_strain(
        self,
        atoms: Atoms,
        max_strain: float = 0.03,
        n_samples: int = 5,
    ) -> List[TrainingData]:
        """
        Generate structures with shear strains.

        Args:
            atoms: Equilibrium structure
            max_strain: Maximum shear strain
            n_samples: Samples per strain component

        Returns:
            List of TrainingData
        """
        results = []

        # Shear strain components: xy, xz, yz
        for component in [(0, 1), (0, 2), (1, 2)]:
            for strain in np.linspace(-max_strain, max_strain, n_samples):
                strained = atoms.copy()
                cell = strained.cell.copy()

                # Apply shear
                i, j = component
                cell[i, j] += strain * cell[j, j]

                strained.set_cell(cell, scale_atoms=True)

                try:
                    data = self._calculate_single(
                        strained,
                        f"shear_{i}{j}_strain_{strain:.3f}"
                    )
                    results.append(data)
                except:
                    pass

        return results

    # ==================== Composition Variation / Subsystem Generation ====================

    def generate_subsystems(
        self,
        atoms: Atoms,
        target_elements: List[str] = None,
    ) -> Dict[str, Atoms]:
        """
        Generate 1-component and 2-component subsystem structures from a base structure.

        For a 3-component system like Nb-W-O, this generates:
        - 1-component: structures with only O, only Nb, only W (elemental/simple oxide)
        - 2-component: Nb-O, W-O, Nb-W systems

        The generated structures maintain the original lattice but replace
        selected elements to create subsystems.

        Args:
            atoms: Base structure (e.g., NbWO)
            target_elements: Elements to consider (default: all unique elements)

        Returns:
            Dict mapping subsystem name to Atoms object
        """
        symbols = atoms.get_chemical_symbols()
        if target_elements is None:
            target_elements = list(set(symbols))

        subsystems = {}

        # 1-component systems (single element)
        for elem in target_elements:
            # Create structure with only this element
            single_atoms = atoms.copy()
            new_symbols = [elem] * len(atoms)
            single_atoms.set_chemical_symbols(new_symbols)
            subsystems[f"1comp_{elem}"] = single_atoms

        # 2-component systems (binary)
        from itertools import combinations
        for elem1, elem2 in combinations(target_elements, 2):
            # Replace elements not in (elem1, elem2) with elem1 or elem2
            binary_atoms = atoms.copy()
            new_symbols = []
            for s in symbols:
                if s == elem1 or s == elem2:
                    new_symbols.append(s)
                else:
                    # Replace with the majority element of the pair
                    count1 = symbols.count(elem1)
                    count2 = symbols.count(elem2)
                    new_symbols.append(elem1 if count1 >= count2 else elem2)
            binary_atoms.set_chemical_symbols(new_symbols)
            subsystems[f"2comp_{elem1}{elem2}"] = binary_atoms

        return subsystems

    def generate_oxide_subsystems(
        self,
        atoms: Atoms,
        metal_elements: List[str],
        oxygen_element: str = "O",
    ) -> Dict[str, Atoms]:
        """
        Generate metal oxide subsystems specifically for oxide materials.

        For NbWO system with metals [Nb, W]:
        - 1-component oxides: Nb-O only, W-O only
        - Pure metals: Nb only, W only (for testing metal-metal interactions)
        - Pure O (for O-O interactions)

        Args:
            atoms: Base oxide structure
            metal_elements: List of metal elements (e.g., ["Nb", "W"])
            oxygen_element: Oxygen element symbol

        Returns:
            Dict mapping subsystem name to Atoms object
        """
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = atoms.cell

        subsystems = {}

        # 1. Single metal oxides (keeping O sites, replacing other metals)
        for target_metal in metal_elements:
            oxide_atoms = atoms.copy()
            new_symbols = []
            for s in symbols:
                if s == oxygen_element:
                    new_symbols.append(oxygen_element)
                elif s == target_metal:
                    new_symbols.append(target_metal)
                else:
                    # Replace other metals with target metal
                    new_symbols.append(target_metal)
            oxide_atoms.set_chemical_symbols(new_symbols)
            subsystems[f"{target_metal}Ox"] = oxide_atoms

        # 2. Pure oxygen (for O-O repulsion)
        pure_O = atoms.copy()
        pure_O.set_chemical_symbols([oxygen_element] * len(atoms))
        subsystems["pure_O"] = pure_O

        # 3. Pure metals (for metal-metal interactions)
        for metal in metal_elements:
            pure_metal = atoms.copy()
            pure_metal.set_chemical_symbols([metal] * len(atoms))
            subsystems[f"pure_{metal}"] = pure_metal

        # 4. Metal alloys (no oxygen)
        if len(metal_elements) >= 2:
            alloy_atoms = atoms.copy()
            new_symbols = []
            metal_idx = 0
            for s in symbols:
                if s == oxygen_element:
                    # Replace O with alternating metals
                    new_symbols.append(metal_elements[metal_idx % len(metal_elements)])
                    metal_idx += 1
                else:
                    new_symbols.append(s)
            alloy_atoms.set_chemical_symbols(new_symbols)
            subsystems["metal_alloy"] = alloy_atoms

        return subsystems

    def sample_subsystems(
        self,
        atoms: Atoms,
        metal_elements: Optional[List[str]] = None,
        oxygen_element: str = "O",
        n_samples_per_system: int = 20,
        include_md: bool = True,
        md_temperatures: List[float] = [300, 600],
        md_steps: int = 200,
    ) -> List[TrainingData]:
        """
        Comprehensively sample subsystem structures.

        For potential fitting, we need data on:
        - Single-element interactions (O-O, Nb-Nb, W-W)
        - Binary interactions (Nb-O, W-O, Nb-W)
        - The full ternary system (Nb-W-O)

        This method generates and samples all subsystems automatically.

        Args:
            atoms: Base structure (3-component)
            metal_elements: Metal elements (auto-detected if None)
            oxygen_element: Oxygen symbol
            n_samples_per_system: Random displacement samples per subsystem
            include_md: Whether to run MD on subsystems
            md_temperatures: MD temperatures
            md_steps: MD steps per temperature

        Returns:
            List of TrainingData from all subsystems
        """
        symbols = atoms.get_chemical_symbols()

        # Auto-detect metals (non-oxygen elements)
        if metal_elements is None:
            all_elements = set(symbols)
            metal_elements = [e for e in all_elements if e != oxygen_element]

        print(f"\n=== Subsystem Sampling ===")
        print(f"  Base system: {atoms.get_chemical_formula()}")
        print(f"  Metals: {metal_elements}")
        print(f"  Oxygen: {oxygen_element}")

        all_data = []

        # Generate subsystems
        subsystems = self.generate_oxide_subsystems(
            atoms, metal_elements, oxygen_element
        )

        print(f"  Generated {len(subsystems)} subsystems:")
        for name in subsystems:
            print(f"    - {name}: {subsystems[name].get_chemical_formula()}")

        # Sample each subsystem
        for name, sub_atoms in subsystems.items():
            print(f"\n  Sampling {name}...")

            # Optimize
            try:
                eq_data = self.optimize_structure(sub_atoms, fmax=0.05)
                eq_data.source = f"subsys_{name}_eq"
                all_data.append(eq_data)
                current = eq_data.structure
            except Exception as e:
                print(f"    Optimization failed: {e}")
                current = sub_atoms.copy()

            # Random displacements
            disp_data = self.sample_random_displacements(
                current, n_samples=n_samples_per_system
            )
            for d in disp_data:
                d.source = f"subsys_{name}_{d.source}"
            all_data.extend(disp_data)

            # Volume strain
            vol_data = self.sample_volume_strain(current, n_samples=5)
            for d in vol_data:
                d.source = f"subsys_{name}_{d.source}"
            all_data.extend(vol_data)

            # MD if requested
            if include_md:
                md_data = self.sample_md(
                    current,
                    temperatures=md_temperatures,
                    n_steps=md_steps,
                    sample_interval=20,
                )
                for d in md_data:
                    d.source = f"subsys_{name}_{d.source}"
                all_data.extend(md_data)

        print(f"\n  Subsystem sampling complete: {len(all_data)} samples")
        return all_data

    # ==================== Site Swap Exploration ====================

    def swap_sites(
        self,
        atoms: Atoms,
        n_swaps: int = 1,
    ) -> Atoms:
        """
        Swap atomic positions between different elements.

        Args:
            atoms: Input structure
            n_swaps: Number of position swaps to perform

        Returns:
            New Atoms with swapped positions
        """
        swapped = atoms.copy()
        symbols = swapped.get_chemical_symbols()
        positions = swapped.get_positions().copy()

        # Get unique elements and their indices
        elements = list(set(symbols))
        if len(elements) < 2:
            return swapped

        for _ in range(n_swaps):
            # Pick two different elements
            elem1, elem2 = np.random.choice(elements, 2, replace=False)

            idx1_list = [i for i, s in enumerate(symbols) if s == elem1]
            idx2_list = [i for i, s in enumerate(symbols) if s == elem2]

            if not idx1_list or not idx2_list:
                continue

            # Random atom from each element
            i1 = np.random.choice(idx1_list)
            i2 = np.random.choice(idx2_list)

            # Swap positions
            positions[i1], positions[i2] = positions[i2].copy(), positions[i1].copy()

        swapped.set_positions(positions)
        return swapped

    def sample_site_swaps(
        self,
        atoms: Atoms,
        n_samples: int = 20,
        max_swaps: int = 3,
        optimize: bool = True,
        fmax: float = 0.05,
    ) -> List[TrainingData]:
        """
        Generate structures by swapping atomic sites.

        Args:
            atoms: Base structure
            n_samples: Number of swap configurations to generate
            max_swaps: Maximum swaps per sample
            optimize: Whether to optimize after swap
            fmax: Force tolerance for optimization

        Returns:
            List of TrainingData
        """
        results = []

        for i in range(n_samples):
            n_swaps = np.random.randint(1, max_swaps + 1)
            swapped = self.swap_sites(atoms, n_swaps=n_swaps)

            try:
                if optimize:
                    swapped.calc = self.calculator
                    opt = LBFGS(swapped, logfile=None)
                    opt.run(fmax=fmax, steps=100)

                data = self._calculate_single(swapped, f"swap_{n_swaps}x_{i}")
                results.append(data)
            except Exception as e:
                warnings.warn(f"Swap sample {i} failed: {e}")

        return results

    def explore_chemical_space(
        self,
        initial_atoms: Atoms,
        n_iterations: int = 10,
        swaps_per_iter: int = 5,
        md_steps: int = 200,
        md_temperature: float = 600,
        collect_interval: int = 20,
    ) -> List[TrainingData]:
        """
        Explore chemical space via iterative swap + MD cycles.

        Algorithm:
        1. Start from initial structure
        2. Generate swap variants
        3. Optimize each variant
        4. Run short MD on promising structures
        5. Collect samples throughout

        Args:
            initial_atoms: Starting structure
            n_iterations: Number of exploration iterations
            swaps_per_iter: Swap samples per iteration
            md_steps: MD steps for sampling
            md_temperature: MD temperature in K
            collect_interval: Steps between MD samples

        Returns:
            List of TrainingData from exploration
        """
        all_data = []

        # Start with optimized initial structure
        print("Optimizing initial structure...")
        try:
            eq_data = self.optimize_structure(initial_atoms)
            all_data.append(eq_data)
            current_best = eq_data.structure.copy()
            best_energy = eq_data.energy
        except Exception as e:
            print(f"Initial optimization failed: {e}")
            current_best = initial_atoms.copy()
            best_energy = float('inf')

        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            # Generate swap variants
            print(f"  Generating {swaps_per_iter} swap variants...")
            swap_data = self.sample_site_swaps(
                current_best,
                n_samples=swaps_per_iter,
                optimize=True,
            )
            all_data.extend(swap_data)

            # Find lowest energy structure from this iteration
            if swap_data:
                energies = [d.energy for d in swap_data]
                min_idx = np.argmin(energies)
                min_energy = energies[min_idx]

                if min_energy < best_energy:
                    best_energy = min_energy
                    current_best = swap_data[min_idx].structure.copy()
                    print(f"  New best energy: {best_energy:.4f} eV")

            # MD sampling on current best
            print(f"  Running MD at {md_temperature}K...")
            md_data = self.sample_md(
                current_best,
                temperatures=[md_temperature],
                n_steps=md_steps,
                sample_interval=collect_interval,
            )
            all_data.extend(md_data)

            print(f"  Total samples: {len(all_data)}")

        return all_data

    # ==================== Combined Sampling ====================

    def collect_comprehensive(
        self,
        atoms: Atoms,
        md_temperatures: List[float] = [300, 600, 1000],
        md_steps: int = 500,
        n_random_disp: int = 50,
        n_volume_samples: int = 11,
        n_swap_samples: int = 20,
        phonon_displacement: float = 0.01,
    ) -> List[TrainingData]:
        """
        Comprehensive data collection combining multiple methods.

        Args:
            atoms: Initial structure (will be optimized first)
            md_temperatures: Temperatures for MD sampling
            md_steps: MD steps per temperature
            n_random_disp: Number of random displacement samples
            n_volume_samples: Number of volume strain samples
            n_swap_samples: Number of site swap samples
            phonon_displacement: Phonon displacement magnitude

        Returns:
            Combined list of TrainingData
        """
        all_data = []

        # 1. Optimize to get equilibrium
        print("Optimizing equilibrium structure...")
        eq_data = self.optimize_structure(atoms)
        all_data.append(eq_data)
        eq_atoms = eq_data.structure

        # 2. Site swaps (explore chemical ordering)
        if n_swap_samples > 0:
            print(f"Sampling {n_swap_samples} site swaps...")
            all_data.extend(self.sample_site_swaps(eq_atoms, n_samples=n_swap_samples))

        # 3. Volume strain (E-V curve)
        print("Sampling volume strains...")
        all_data.extend(self.sample_volume_strain(eq_atoms, n_samples=n_volume_samples))

        # 4. Shear strains
        print("Sampling shear strains...")
        all_data.extend(self.sample_shear_strain(eq_atoms))

        # 5. Random displacements
        print("Sampling random displacements...")
        all_data.extend(self.sample_random_displacements(eq_atoms, n_samples=n_random_disp))

        # 6. MD sampling
        print(f"Running MD at temperatures {md_temperatures}...")
        all_data.extend(self.sample_md(eq_atoms, temperatures=md_temperatures, n_steps=md_steps))

        # 7. Phonon displacements (small structures only)
        if len(atoms) <= 20:
            print("Sampling phonon displacements...")
            all_data.extend(self.sample_phonon_displacements(eq_atoms, displacement=phonon_displacement))

        print(f"Total samples collected: {len(all_data)}")
        return all_data

    def collect_from_initial_guesses(
        self,
        initial_structures: List[Atoms],
        n_iterations: int = 5,
        swaps_per_iter: int = 10,
        md_temperatures: List[float] = [300, 600, 1000],
        md_steps_per_temp: int = 200,
        n_volume_samples: int = 7,
        n_random_disp: int = 30,
        include_subsystems: bool = False,
        metal_elements: Optional[List[str]] = None,
        use_diverse_md: bool = False,
    ) -> List[TrainingData]:
        """
        Full pipeline: start from initial guesses, explore via swaps/MD.

        Pipeline per structure:
        1. Optimize initial guess
        2. Site swap exploration (iterate)
        3. Volume/shear strain sampling
        4. Random displacements
        5. MD at multiple temperatures
        6. (Optional) Subsystem sampling for 1/2-component systems

        Args:
            initial_structures: List of initial structure guesses
            n_iterations: Swap exploration iterations per structure
            swaps_per_iter: Swap samples per iteration
            md_temperatures: MD sampling temperatures
            md_steps_per_temp: MD steps per temperature
            n_volume_samples: Volume strain samples
            n_random_disp: Random displacement samples
            include_subsystems: Sample 1/2-component subsystems
            metal_elements: Metal elements for subsystem generation
            use_diverse_md: Use enhanced MD sampling (more temps, trajectories)

        Returns:
            Combined TrainingData from all structures
        """
        all_data = []

        for struct_idx, atoms in enumerate(initial_structures):
            formula = atoms.get_chemical_formula()
            print(f"\n{'='*60}")
            print(f"Structure {struct_idx + 1}/{len(initial_structures)}: {formula}")
            print(f"{'='*60}")

            # Phase 1: Optimize initial
            print("\n[Phase 1] Optimizing initial structure...")
            try:
                eq_data = self.optimize_structure(atoms)
                all_data.append(eq_data)
                current = eq_data.structure
            except Exception as e:
                print(f"  Optimization failed: {e}, using original")
                current = atoms.copy()

            # Phase 2: Swap exploration
            print(f"\n[Phase 2] Site swap exploration ({n_iterations} iterations)...")
            for it in range(n_iterations):
                swap_data = self.sample_site_swaps(
                    current,
                    n_samples=swaps_per_iter,
                    optimize=True,
                )
                all_data.extend(swap_data)

                # Update current to best structure
                if swap_data:
                    energies = [d.energy for d in swap_data]
                    best_idx = np.argmin(energies)
                    current = swap_data[best_idx].structure.copy()
                    print(f"  Iter {it+1}: {len(swap_data)} samples, best E={energies[best_idx]:.4f} eV")

            # Phase 3: Strain sampling on best structure
            print(f"\n[Phase 3] Strain sampling...")
            all_data.extend(self.sample_volume_strain(current, n_samples=n_volume_samples))
            all_data.extend(self.sample_shear_strain(current))

            # Phase 4: Random displacements
            print(f"\n[Phase 4] Random displacements...")
            all_data.extend(self.sample_random_displacements(current, n_samples=n_random_disp))

            # Phase 5: MD sampling
            if use_diverse_md:
                print(f"\n[Phase 5] Diverse MD sampling...")
                all_data.extend(self.sample_md_diverse(
                    current,
                    temperature_range=(100, 1200),
                    n_temperature_points=8,
                    n_trajectories_per_temp=2,
                    n_steps=300,
                    sample_interval=20,
                ))
            else:
                print(f"\n[Phase 5] MD sampling at {md_temperatures}K...")
                all_data.extend(self.sample_md(
                    current,
                    temperatures=md_temperatures,
                    n_steps=md_steps_per_temp,
                    sample_interval=20,
                ))

            # Phase 6: Subsystem sampling (1-component, 2-component)
            if include_subsystems:
                print(f"\n[Phase 6] Subsystem sampling (1/2-component systems)...")
                subsys_data = self.sample_subsystems(
                    current,
                    metal_elements=metal_elements,
                    n_samples_per_system=15,
                    include_md=True,
                    md_temperatures=[300, 600],
                    md_steps=150,
                )
                all_data.extend(subsys_data)

            print(f"\nStructure complete. Running total: {len(all_data)} samples")

        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE: {len(all_data)} total samples")
        print(f"{'='*60}")

        return all_data

    def collect_with_composition_diversity(
        self,
        base_structures: List[Atoms],
        metal_elements: List[str],
        oxygen_element: str = "O",
        n_swap_iterations: int = 3,
        swaps_per_iter: int = 8,
        n_random_disp: int = 20,
        use_diverse_md: bool = True,
    ) -> List[TrainingData]:
        """
        Comprehensive data collection with composition diversity.

        This method automatically samples:
        1. The base 3-component structures (e.g., NbWO)
        2. All 2-component subsystems (e.g., NbO, WO, NbW)
        3. All 1-component systems (e.g., pure Nb, W, O)

        This ensures the fitted potential has good coverage of all
        pairwise interactions.

        Args:
            base_structures: Initial 3-component structures
            metal_elements: List of metal elements (e.g., ["Nb", "W"])
            oxygen_element: Oxygen element symbol
            n_swap_iterations: Site swap iterations per structure
            swaps_per_iter: Swap samples per iteration
            n_random_disp: Random displacement samples per system
            use_diverse_md: Use enhanced MD sampling

        Returns:
            List of TrainingData covering all compositions
        """
        all_data = []

        print("\n" + "=" * 70)
        print("COMPREHENSIVE DATA COLLECTION WITH COMPOSITION DIVERSITY")
        print("=" * 70)
        print(f"Base structures: {len(base_structures)}")
        print(f"Metal elements: {metal_elements}")
        print(f"Oxygen: {oxygen_element}")
        print("=" * 70)

        # Phase 1: 3-component (base) structures
        print("\n### PHASE 1: 3-Component Structures ###")
        for i, atoms in enumerate(base_structures):
            print(f"\n[{i+1}/{len(base_structures)}] {atoms.get_chemical_formula()}")

            # Optimize
            try:
                eq_data = self.optimize_structure(atoms)
                all_data.append(eq_data)
                current = eq_data.structure
            except:
                current = atoms.copy()

            # Site swaps
            for it in range(n_swap_iterations):
                swap_data = self.sample_site_swaps(current, n_samples=swaps_per_iter)
                all_data.extend(swap_data)
                if swap_data:
                    best_idx = np.argmin([d.energy for d in swap_data])
                    current = swap_data[best_idx].structure.copy()

            # Strain
            all_data.extend(self.sample_volume_strain(current, n_samples=7))
            all_data.extend(self.sample_shear_strain(current, n_samples=3))

            # Random displacements
            all_data.extend(self.sample_random_displacements(current, n_samples=n_random_disp))

            # MD
            if use_diverse_md:
                all_data.extend(self.sample_md_diverse(
                    current,
                    temperature_range=(100, 1200),
                    n_temperature_points=6,
                    n_trajectories_per_temp=2,
                    n_steps=200,
                ))
            else:
                all_data.extend(self.sample_md(
                    current,
                    temperatures=[300, 600, 1000],
                    n_steps=300,
                ))

        # Phase 2: 2-component subsystems
        print("\n### PHASE 2: 2-Component Subsystems ###")
        # Use first base structure as template
        template = base_structures[0]

        # Generate binary oxide subsystems
        for metal in metal_elements:
            print(f"\n[2-comp] {metal}-O binary oxide...")
            binary = template.copy()
            symbols = template.get_chemical_symbols()
            new_symbols = []
            for s in symbols:
                if s == oxygen_element:
                    new_symbols.append(oxygen_element)
                else:
                    new_symbols.append(metal)
            binary.set_chemical_symbols(new_symbols)

            try:
                eq_data = self.optimize_structure(binary)
                eq_data.source = f"2comp_{metal}O_eq"
                all_data.append(eq_data)
                current = eq_data.structure
            except:
                current = binary.copy()

            all_data.extend(self.sample_random_displacements(current, n_samples=n_random_disp))
            all_data.extend(self.sample_volume_strain(current, n_samples=5))

            if use_diverse_md:
                md_data = self.sample_md_diverse(
                    current,
                    temperature_range=(100, 1000),
                    n_temperature_points=5,
                    n_trajectories_per_temp=2,
                    n_steps=150,
                    heating_cooling_ramps=False,
                )
            else:
                md_data = self.sample_md(current, temperatures=[300, 600], n_steps=200)

            for d in md_data:
                d.source = f"2comp_{metal}O_{d.source}"
            all_data.extend(md_data)

        # Metal alloy (no oxygen)
        if len(metal_elements) >= 2:
            print(f"\n[2-comp] Metal alloy ({'-'.join(metal_elements)})...")
            alloy = template.copy()
            symbols = template.get_chemical_symbols()
            new_symbols = []
            metal_idx = 0
            for s in symbols:
                if s == oxygen_element:
                    new_symbols.append(metal_elements[metal_idx % len(metal_elements)])
                    metal_idx += 1
                else:
                    new_symbols.append(s)
            alloy.set_chemical_symbols(new_symbols)

            try:
                eq_data = self.optimize_structure(alloy)
                eq_data.source = "2comp_alloy_eq"
                all_data.append(eq_data)
                current = eq_data.structure
            except:
                current = alloy.copy()

            all_data.extend(self.sample_random_displacements(current, n_samples=n_random_disp))
            all_data.extend(self.sample_volume_strain(current, n_samples=5))
            md_data = self.sample_md(current, temperatures=[300, 600], n_steps=150)
            for d in md_data:
                d.source = f"2comp_alloy_{d.source}"
            all_data.extend(md_data)

        # Phase 3: 1-component systems
        print("\n### PHASE 3: 1-Component Systems ###")

        # Pure oxygen
        print(f"\n[1-comp] Pure {oxygen_element}...")
        pure_O = template.copy()
        pure_O.set_chemical_symbols([oxygen_element] * len(template))
        try:
            eq_data = self.optimize_structure(pure_O)
            eq_data.source = "1comp_O_eq"
            all_data.append(eq_data)
            current = eq_data.structure
        except:
            current = pure_O.copy()

        all_data.extend(self.sample_random_displacements(current, n_samples=n_random_disp // 2))
        all_data.extend(self.sample_volume_strain(current, n_samples=5))
        md_data = self.sample_md(current, temperatures=[300, 500], n_steps=100)
        for d in md_data:
            d.source = f"1comp_O_{d.source}"
        all_data.extend(md_data)

        # Pure metals
        for metal in metal_elements:
            print(f"\n[1-comp] Pure {metal}...")
            pure_metal = template.copy()
            pure_metal.set_chemical_symbols([metal] * len(template))
            try:
                eq_data = self.optimize_structure(pure_metal)
                eq_data.source = f"1comp_{metal}_eq"
                all_data.append(eq_data)
                current = eq_data.structure
            except:
                current = pure_metal.copy()

            all_data.extend(self.sample_random_displacements(current, n_samples=n_random_disp // 2))
            all_data.extend(self.sample_volume_strain(current, n_samples=5))
            md_data = self.sample_md(current, temperatures=[300, 600], n_steps=100)
            for d in md_data:
                d.source = f"1comp_{metal}_{d.source}"
            all_data.extend(md_data)

        print("\n" + "=" * 70)
        print(f"COLLECTION COMPLETE: {len(all_data)} total samples")
        print("=" * 70)

        return all_data

    # ==================== Export Functions ====================

    def save_gulp_format(
        self,
        data: List[TrainingData],
        output_file: str,
        species_charges: Dict[str, float],
        potential_template: str = "",
    ) -> None:
        """
        Save training data in GULP fitting format.

        Args:
            data: List of TrainingData
            output_file: Output .gin file path
            species_charges: Element -> charge mapping
            potential_template: Initial potential definition
        """
        lines = ["fit conp\n"]

        # Species
        lines.append("species")
        for elem, charge in species_charges.items():
            lines.append(f"{elem} core {charge:.4f}")
        lines.append("")

        # Each structure as observable
        for i, d in enumerate(data):
            atoms = d.structure
            cell = atoms.cell
            a, b, c = cell.lengths()
            alpha, beta, gamma = cell.angles()

            lines.append(f"# Structure {i}: {d.source}")
            lines.append("cell")
            lines.append(f"{a:.10f} {b:.10f} {c:.10f} {alpha:.6f} {beta:.6f} {gamma:.6f}")
            lines.append("")

            lines.append("frac")
            scaled = atoms.get_scaled_positions()
            symbols = atoms.get_chemical_symbols()
            for sym, pos in zip(symbols, scaled):
                lines.append(f"{sym} core {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
            lines.append("")

            # Observable: energy
            lines.append("observables")
            lines.append("energy ev")
            lines.append(f"{d.energy:.10f} 1.0")
            lines.append("end")
            lines.append("")

        # Potential template
        if potential_template:
            lines.append(potential_template)

        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        print(f"Saved {len(data)} structures to {output_file}")

    def save_json(
        self,
        data: List[TrainingData],
        output_file: str,
    ) -> None:
        """Save training data as JSON for flexibility."""
        records = []
        for d in data:
            record = {
                "energy": d.energy,
                "forces": d.forces.tolist(),
                "stress": d.stress.tolist() if d.stress is not None else None,
                "source": d.source,
                "cell": d.structure.cell.tolist(),
                "positions": d.structure.positions.tolist(),
                "symbols": d.structure.get_chemical_symbols(),
                "pbc": d.structure.pbc.tolist(),
            }
            records.append(record)

        with open(output_file, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Saved {len(data)} structures to {output_file}")

    def save_xyz(
        self,
        data: List[TrainingData],
        output_file: str,
    ) -> None:
        """Save structures as extended XYZ with energy/forces."""
        atoms_list = []
        for d in data:
            atoms = d.structure.copy()
            atoms.info["energy"] = d.energy
            atoms.info["source"] = d.source
            atoms.arrays["forces"] = d.forces
            atoms_list.append(atoms)

        write(output_file, atoms_list, format="extxyz")
        print(f"Saved {len(data)} structures to {output_file}")


def collect_from_cif_files(
    cif_dir: str,
    calculator: Calculator,
    output_prefix: str = "training_data",
    species_charges: Optional[Dict[str, float]] = None,
) -> List[TrainingData]:
    """
    Convenience function to collect data from CIF files.

    Args:
        cif_dir: Directory containing CIF files
        calculator: MLIP calculator
        output_prefix: Prefix for output files
        species_charges: Element charges for GULP format

    Returns:
        List of TrainingData
    """
    from glob import glob

    collector = DataCollector(calculator)
    all_data = []

    cif_files = glob(os.path.join(cif_dir, "*.cif"))
    print(f"Found {len(cif_files)} CIF files")

    for cif_file in cif_files:
        print(f"\nProcessing {os.path.basename(cif_file)}...")
        try:
            atoms = read(cif_file)
            data = collector.collect_comprehensive(atoms)
            all_data.extend(data)
        except Exception as e:
            print(f"  Error: {e}")

    # Save in multiple formats
    collector.save_json(all_data, f"{output_prefix}.json")
    collector.save_xyz(all_data, f"{output_prefix}.xyz")

    if species_charges:
        collector.save_gulp_format(all_data, f"{output_prefix}.gin", species_charges)

    return all_data
