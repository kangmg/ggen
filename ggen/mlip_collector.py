"""
MLIP-based Data Collector with ASE Database Storage.

Collects training data using Machine Learning Interatomic Potentials (MLIP)
and stores ALL computed structures to an ASE database for later use in
training classical potentials like GULP.

Features:
- MD trajectory collection at various temperatures
- Geometry optimization trajectory collection
- Structure rattling/perturbation sampling
- NPT ensemble sampling for pressure effects
- Subsystem generation (1/2/3-component)
- Complete storage to ASE database with metadata
"""

import os
import time
import hashlib
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from enum import Enum
from datetime import datetime
import json
import warnings

import numpy as np
from ase import Atoms, units
from ase.db import connect
from ase.io import read, write, Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.filters import ExpCellFilter, StrainFilter
from ase.calculators.calculator import Calculator, PropertyNotImplementedError


class CollectionMode(Enum):
    """Data collection mode."""
    MD_NVT = "md_nvt"
    MD_NPT = "md_npt"
    MD_LANGEVIN = "md_langevin"
    OPTIMIZATION = "optimization"
    CELL_OPTIMIZATION = "cell_optimization"
    SINGLE_POINT = "single_point"
    RATTLING = "rattling"
    STRAIN = "strain"
    HEATING_RAMP = "heating_ramp"
    COOLING_RAMP = "cooling_ramp"


@dataclass
class CollectionConfig:
    """Configuration for data collection."""
    # Database settings
    db_path: str = "mlip_training_data.db"

    # MD settings
    md_temperatures: List[float] = field(default_factory=lambda: [100, 300, 500, 800, 1000, 1500])
    md_steps_per_temp: int = 500
    md_timestep: float = 1.0  # fs
    md_friction: float = 0.01  # 1/fs for Langevin
    md_save_interval: int = 10  # Save every N steps

    # NPT settings
    npt_pressures: List[float] = field(default_factory=lambda: [0.0, 1.0, 5.0, 10.0])  # GPa
    npt_ttime: float = 25.0  # Temperature coupling time (fs)
    npt_ptime: float = 100.0  # Pressure coupling time (fs)

    # Optimization settings
    opt_fmax: float = 0.05  # eV/Å
    opt_max_steps: int = 200
    opt_save_interval: int = 5

    # Rattling settings
    rattle_stdev: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    rattle_n_configs: int = 10  # Per stdev value

    # Strain settings
    strain_magnitudes: List[float] = field(default_factory=lambda: [-0.05, -0.02, 0.02, 0.05])

    # Heating/cooling ramp
    ramp_start_temp: float = 100.0
    ramp_end_temp: float = 1500.0
    ramp_steps: int = 1000

    # General
    seed: int = 42
    verbose: bool = True


@dataclass
class CollectionStats:
    """Statistics for data collection run."""
    total_structures: int = 0
    md_structures: int = 0
    opt_structures: int = 0
    rattle_structures: int = 0
    strain_structures: int = 0
    single_point_structures: int = 0
    failed_calculations: int = 0
    collection_time: float = 0.0

    def summary(self) -> str:
        return f"""
Collection Statistics:
  Total structures: {self.total_structures}
  - MD: {self.md_structures}
  - Optimization: {self.opt_structures}
  - Rattling: {self.rattle_structures}
  - Strain: {self.strain_structures}
  - Single point: {self.single_point_structures}
  Failed: {self.failed_calculations}
  Time: {self.collection_time:.1f}s
"""


class MLIPCollector:
    """
    MLIP-based data collector with ASE database storage.

    Collects diverse training data using MLIP calculations and stores
    everything to an ASE database for later GULP potential fitting.

    Example:
        from ase.calculators.emt import EMT  # or your MLIP calculator

        collector = MLIPCollector(
            calculator=EMT(),
            db_path="training_data.db"
        )

        # Collect from initial structures
        structures = [read("structure1.cif"), read("structure2.cif")]
        stats = collector.collect_all(structures, name_prefix="NbWO")

        # Query collected data
        data = collector.get_training_data(max_force=10.0)
    """

    def __init__(
        self,
        calculator: Calculator,
        db_path: str = "mlip_training_data.db",
        config: Optional[CollectionConfig] = None,
    ):
        """
        Initialize MLIP collector.

        Args:
            calculator: ASE calculator (MLIP, ORB, MACE, etc.)
            db_path: Path to ASE database
            config: Collection configuration
        """
        self.calculator = calculator
        self.db_path = db_path
        self.config = config or CollectionConfig(db_path=db_path)

        # Initialize database
        self.db = connect(self.db_path)

        # Statistics
        self.stats = CollectionStats()

        # RNG
        self.rng = np.random.default_rng(self.config.seed)

        self._log(f"MLIPCollector initialized with database: {self.db_path}")

    def _log(self, msg: str) -> None:
        """Print log message if verbose."""
        if self.config.verbose:
            print(f"[MLIPCollector] {msg}")

    def _get_structure_hash(self, atoms: Atoms) -> str:
        """Generate hash for structure identification."""
        data = (
            atoms.get_chemical_formula() +
            str(np.round(atoms.get_positions(), 4).tobytes()) +
            str(np.round(atoms.get_cell(), 4).tobytes())
        )
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def _calculate_and_store(
        self,
        atoms: Atoms,
        mode: CollectionMode,
        source_name: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Calculate properties and store to database.

        Args:
            atoms: Structure to calculate
            mode: Collection mode
            source_name: Source structure identifier
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Attach calculator
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.calculator

            # Calculate properties
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()

            try:
                stress = atoms_copy.get_stress()
            except (RuntimeError, PropertyNotImplementedError):
                stress = None

            # Build key-value data
            # Note: 'energy' is reserved in ASE db, use 'total_energy'
            kvp = {
                "mode": mode.value,
                "source": source_name,
                "total_energy": energy,
                "max_force": float(np.max(np.abs(forces))),
                "mean_force": float(np.mean(np.linalg.norm(forces, axis=1))),
                "timestamp": datetime.now().isoformat(),
                "structure_hash": self._get_structure_hash(atoms_copy),
            }

            if stress is not None:
                # Stress can be Voigt (6,) or full (3,3)
                if stress.shape == (6,):
                    # Voigt notation: xx, yy, zz, yz, xz, xy
                    kvp["stress_trace"] = float((stress[0] + stress[1] + stress[2]) / 3)
                else:
                    kvp["stress_trace"] = float(np.trace(stress.reshape(3, 3)) / 3)

            # Add metadata
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (int, float, str, bool)):
                        kvp[k] = v

            # Store forces and stress as data (arrays)
            data = {
                "forces": forces,
            }
            if stress is not None:
                data["stress"] = stress

            # Write to database
            self.db.write(atoms_copy, key_value_pairs=kvp, data=data)

            self.stats.total_structures += 1
            return True

        except Exception as e:
            self._log(f"Calculation failed: {e}")
            self.stats.failed_calculations += 1
            return False

    def collect_single_point(
        self,
        atoms: Atoms,
        name: str = "structure",
    ) -> bool:
        """Collect single point calculation."""
        self._log(f"Single point: {name}")
        success = self._calculate_and_store(
            atoms, CollectionMode.SINGLE_POINT, name
        )
        if success:
            self.stats.single_point_structures += 1
        return success

    def collect_md_trajectory(
        self,
        atoms: Atoms,
        temperature: float,
        n_steps: int,
        name: str = "md",
        ensemble: str = "langevin",
        pressure: Optional[float] = None,
        save_interval: Optional[int] = None,
    ) -> int:
        """
        Collect MD trajectory at given temperature.

        Args:
            atoms: Initial structure
            temperature: Temperature in K
            n_steps: Number of MD steps
            name: Trajectory identifier
            ensemble: "langevin", "nvt", or "npt"
            pressure: Pressure in GPa (for NPT)
            save_interval: Save every N steps

        Returns:
            Number of structures collected
        """
        save_interval = save_interval or self.config.md_save_interval

        self._log(f"MD {ensemble} at {temperature}K for {n_steps} steps: {name}")

        # Prepare atoms
        atoms_md = atoms.copy()
        atoms_md.calc = self.calculator

        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms_md, temperature_K=temperature)

        # Create dynamics
        timestep = self.config.md_timestep * units.fs

        if ensemble == "langevin":
            dyn = Langevin(
                atoms_md,
                timestep,
                temperature_K=temperature,
                friction=self.config.md_friction,
            )
        elif ensemble == "nvt":
            dyn = NVTBerendsen(
                atoms_md,
                timestep,
                temperature_K=temperature,
                taut=self.config.npt_ttime * units.fs,
            )
        elif ensemble == "npt" and pressure is not None:
            # NPT with pressure
            dyn = NPT(
                atoms_md,
                timestep,
                temperature_K=temperature,
                externalstress=pressure * units.GPa,
                ttime=self.config.npt_ttime * units.fs,
                pfactor=self.config.npt_ptime * units.fs,
            )
        else:
            dyn = Langevin(
                atoms_md,
                timestep,
                temperature_K=temperature,
                friction=self.config.md_friction,
            )

        # Collection counter
        collected = 0

        def store_frame():
            nonlocal collected
            metadata = {
                "temperature": temperature,
                "md_step": dyn.nsteps,
                "ensemble": ensemble,
            }
            if pressure is not None:
                metadata["pressure_gpa"] = pressure

            mode = CollectionMode.MD_NPT if ensemble == "npt" else CollectionMode.MD_LANGEVIN

            if self._calculate_and_store(atoms_md, mode, name, metadata):
                collected += 1
                self.stats.md_structures += 1

        # Store initial
        store_frame()

        # Run MD with collection
        for step in range(n_steps):
            try:
                dyn.run(1)
                if (step + 1) % save_interval == 0:
                    store_frame()
            except Exception as e:
                self._log(f"MD step {step} failed: {e}")
                break

        self._log(f"  Collected {collected} frames")
        return collected

    def collect_md_temperatures(
        self,
        atoms: Atoms,
        name: str = "md_temp",
        temperatures: Optional[List[float]] = None,
        n_steps: Optional[int] = None,
    ) -> int:
        """
        Collect MD trajectories at multiple temperatures.

        Args:
            atoms: Initial structure
            name: Base name for trajectories
            temperatures: List of temperatures (K)
            n_steps: Steps per temperature

        Returns:
            Total structures collected
        """
        temperatures = temperatures or self.config.md_temperatures
        n_steps = n_steps or self.config.md_steps_per_temp

        total = 0
        for temp in temperatures:
            traj_name = f"{name}_T{temp}K"
            total += self.collect_md_trajectory(
                atoms, temp, n_steps, traj_name
            )

        return total

    def collect_heating_ramp(
        self,
        atoms: Atoms,
        name: str = "heating",
        start_temp: Optional[float] = None,
        end_temp: Optional[float] = None,
        n_steps: Optional[int] = None,
    ) -> int:
        """Collect heating ramp trajectory."""
        start_temp = start_temp or self.config.ramp_start_temp
        end_temp = end_temp or self.config.ramp_end_temp
        n_steps = n_steps or self.config.ramp_steps

        self._log(f"Heating ramp {start_temp}K -> {end_temp}K: {name}")

        atoms_md = atoms.copy()
        atoms_md.calc = self.calculator

        MaxwellBoltzmannDistribution(atoms_md, temperature_K=start_temp)

        timestep = self.config.md_timestep * units.fs
        temp_increment = (end_temp - start_temp) / n_steps

        collected = 0
        current_temp = start_temp

        dyn = Langevin(
            atoms_md, timestep,
            temperature_K=current_temp,
            friction=self.config.md_friction,
        )

        save_interval = max(1, n_steps // 100)  # ~100 frames

        for step in range(n_steps):
            try:
                # Update temperature
                current_temp = start_temp + temp_increment * step
                dyn.set_temperature(temperature_K=current_temp)

                dyn.run(1)

                if step % save_interval == 0:
                    metadata = {
                        "temperature": current_temp,
                        "md_step": step,
                        "ramp_progress": step / n_steps,
                    }
                    if self._calculate_and_store(
                        atoms_md, CollectionMode.HEATING_RAMP, name, metadata
                    ):
                        collected += 1
                        self.stats.md_structures += 1

            except Exception as e:
                self._log(f"Heating step {step} failed: {e}")
                break

        self._log(f"  Collected {collected} frames")
        return collected

    def collect_cooling_ramp(
        self,
        atoms: Atoms,
        name: str = "cooling",
        start_temp: Optional[float] = None,
        end_temp: Optional[float] = None,
        n_steps: Optional[int] = None,
    ) -> int:
        """Collect cooling ramp trajectory (quenching)."""
        start_temp = start_temp or self.config.ramp_end_temp
        end_temp = end_temp or self.config.ramp_start_temp

        return self.collect_heating_ramp(
            atoms, name, start_temp, end_temp, n_steps
        )

    def collect_optimization(
        self,
        atoms: Atoms,
        name: str = "opt",
        optimize_cell: bool = False,
        optimizer: str = "BFGS",
    ) -> int:
        """
        Collect optimization trajectory.

        Args:
            atoms: Initial structure
            name: Optimization identifier
            optimize_cell: Also optimize cell parameters
            optimizer: Optimizer type ("BFGS", "FIRE", "LBFGS")

        Returns:
            Number of structures collected
        """
        mode = CollectionMode.CELL_OPTIMIZATION if optimize_cell else CollectionMode.OPTIMIZATION
        self._log(f"Optimization {'(cell)' if optimize_cell else ''}: {name}")

        atoms_opt = atoms.copy()
        atoms_opt.calc = self.calculator

        # Apply cell filter if needed
        if optimize_cell:
            atoms_opt = ExpCellFilter(atoms_opt)

        # Select optimizer
        opt_classes = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}
        opt_class = opt_classes.get(optimizer, BFGS)

        collected = 0
        step_count = [0]

        def store_step():
            nonlocal collected
            step_count[0] += 1

            if step_count[0] % self.config.opt_save_interval == 0:
                # Get actual atoms (unwrap filter if present)
                actual_atoms = atoms_opt.atoms if hasattr(atoms_opt, 'atoms') else atoms_opt

                metadata = {
                    "opt_step": step_count[0],
                    "optimizer": optimizer,
                    "optimize_cell": optimize_cell,
                }

                if self._calculate_and_store(actual_atoms, mode, name, metadata):
                    collected += 1
                    self.stats.opt_structures += 1

        # Create optimizer
        with tempfile.NamedTemporaryFile(suffix='.log', delete=True) as f:
            opt = opt_class(atoms_opt, logfile=f.name)
            opt.attach(store_step)

            try:
                opt.run(fmax=self.config.opt_fmax, steps=self.config.opt_max_steps)
            except Exception as e:
                self._log(f"Optimization failed: {e}")

        # Store final structure
        actual_atoms = atoms_opt.atoms if hasattr(atoms_opt, 'atoms') else atoms_opt
        metadata = {"opt_step": step_count[0], "is_final": True, "optimizer": optimizer}
        if self._calculate_and_store(actual_atoms, mode, name, metadata):
            collected += 1
            self.stats.opt_structures += 1

        self._log(f"  Collected {collected} frames")
        return collected

    def collect_rattling(
        self,
        atoms: Atoms,
        name: str = "rattle",
        stdevs: Optional[List[float]] = None,
        n_configs: Optional[int] = None,
    ) -> int:
        """
        Collect rattled/perturbed structures.

        Args:
            atoms: Base structure
            name: Identifier
            stdevs: Standard deviations for rattling (Å)
            n_configs: Number of configurations per stdev

        Returns:
            Number of structures collected
        """
        stdevs = stdevs or self.config.rattle_stdev
        n_configs = n_configs or self.config.rattle_n_configs

        self._log(f"Rattling with stdevs={stdevs}: {name}")

        collected = 0

        for stdev in stdevs:
            for i in range(n_configs):
                atoms_rattled = atoms.copy()

                # Add random displacements
                displacements = self.rng.normal(0, stdev, atoms_rattled.positions.shape)
                atoms_rattled.positions += displacements

                metadata = {
                    "rattle_stdev": stdev,
                    "rattle_config": i,
                }

                if self._calculate_and_store(
                    atoms_rattled, CollectionMode.RATTLING, name, metadata
                ):
                    collected += 1
                    self.stats.rattle_structures += 1

        self._log(f"  Collected {collected} structures")
        return collected

    def collect_strain(
        self,
        atoms: Atoms,
        name: str = "strain",
        strains: Optional[List[float]] = None,
    ) -> int:
        """
        Collect strained structures.

        Args:
            atoms: Base structure
            name: Identifier
            strains: Strain magnitudes (fractional)

        Returns:
            Number of structures collected
        """
        strains = strains or self.config.strain_magnitudes

        self._log(f"Strain sampling: {name}")

        collected = 0

        # Volumetric strain
        for strain in strains:
            atoms_strained = atoms.copy()
            atoms_strained.set_cell(
                atoms_strained.get_cell() * (1 + strain),
                scale_atoms=True
            )

            metadata = {
                "strain_type": "volumetric",
                "strain_magnitude": strain,
            }

            if self._calculate_and_store(
                atoms_strained, CollectionMode.STRAIN, name, metadata
            ):
                collected += 1
                self.stats.strain_structures += 1

        # Uniaxial strains (x, y, z)
        for axis in range(3):
            for strain in strains:
                atoms_strained = atoms.copy()
                cell = atoms_strained.get_cell().copy()
                cell[axis] *= (1 + strain)
                atoms_strained.set_cell(cell, scale_atoms=True)

                metadata = {
                    "strain_type": f"uniaxial_{['x', 'y', 'z'][axis]}",
                    "strain_magnitude": strain,
                }

                if self._calculate_and_store(
                    atoms_strained, CollectionMode.STRAIN, name, metadata
                ):
                    collected += 1
                    self.stats.strain_structures += 1

        self._log(f"  Collected {collected} structures")
        return collected

    def collect_npt_pressures(
        self,
        atoms: Atoms,
        name: str = "npt",
        temperature: float = 300.0,
        pressures: Optional[List[float]] = None,
        n_steps: Optional[int] = None,
    ) -> int:
        """
        Collect NPT trajectories at various pressures.

        Args:
            atoms: Initial structure
            name: Base name
            temperature: Temperature in K
            pressures: Pressures in GPa
            n_steps: Steps per pressure

        Returns:
            Total structures collected
        """
        pressures = pressures or self.config.npt_pressures
        n_steps = n_steps or self.config.md_steps_per_temp

        total = 0
        for pressure in pressures:
            traj_name = f"{name}_P{pressure}GPa"
            total += self.collect_md_trajectory(
                atoms, temperature, n_steps, traj_name,
                ensemble="npt", pressure=pressure
            )

        return total

    def generate_subsystems(
        self,
        atoms: Atoms,
        metal_elements: List[str],
        oxygen_element: str = "O",
    ) -> List[Tuple[str, Atoms]]:
        """
        Generate 1-component and 2-component subsystems.

        Args:
            atoms: Base 3-component structure
            metal_elements: List of metal elements (e.g., ["Nb", "W"])
            oxygen_element: Oxygen element symbol

        Returns:
            List of (name, atoms) tuples for subsystems
        """
        subsystems = []
        cell = atoms.get_cell()

        # 2-component: each metal + oxygen
        for metal in metal_elements:
            # Extract metal and oxygen positions
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()

            indices = [i for i, s in enumerate(symbols) if s in [metal, oxygen_element]]

            if len(indices) > 1:
                sub_atoms = Atoms(
                    symbols=[symbols[i] for i in indices],
                    positions=[positions[i] for i in indices],
                    cell=cell,
                    pbc=True,
                )
                subsystems.append((f"{metal}O_2comp", sub_atoms))

        # 1-component: pure metals (simple structures)
        for metal in metal_elements:
            # Create simple BCC structure
            from ase.build import bulk
            try:
                metal_bulk = bulk(metal, 'bcc', a=3.3, cubic=True)
                subsystems.append((f"{metal}_1comp", metal_bulk))
            except (ValueError, KeyError, RuntimeError) as e:
                # Skip metals that don't have lattice parameters defined
                warnings.warn(f"Could not create bulk structure for {metal}: {e}")

        # 1-component: oxygen (hypothetical, for completeness)
        # Usually not needed for oxide potentials

        return subsystems

    def collect_all(
        self,
        structures: List[Atoms],
        name_prefix: str = "struct",
        include_md: bool = True,
        include_opt: bool = True,
        include_rattling: bool = True,
        include_strain: bool = True,
        include_heating_cooling: bool = True,
        include_npt: bool = False,
        include_subsystems: bool = False,
        metal_elements: Optional[List[str]] = None,
    ) -> CollectionStats:
        """
        Comprehensive data collection from initial structures.

        Args:
            structures: List of initial structures
            name_prefix: Prefix for naming
            include_md: Include MD trajectories
            include_opt: Include optimization trajectories
            include_rattling: Include rattled structures
            include_strain: Include strained structures
            include_heating_cooling: Include heating/cooling ramps
            include_npt: Include NPT sampling
            include_subsystems: Include 1/2-component subsystems
            metal_elements: Metal elements for subsystem generation

        Returns:
            Collection statistics
        """
        start_time = time.time()

        self._log(f"Starting comprehensive collection from {len(structures)} structures")

        for i, atoms in enumerate(structures):
            struct_name = f"{name_prefix}_{i}"
            formula = atoms.get_chemical_formula()
            self._log(f"\n=== Structure {i+1}/{len(structures)}: {formula} ===")

            # Single point on initial
            self.collect_single_point(atoms, f"{struct_name}_initial")

            # Optimization
            if include_opt:
                self.collect_optimization(atoms, f"{struct_name}_opt", optimize_cell=False)
                self.collect_optimization(atoms, f"{struct_name}_cellopt", optimize_cell=True)

            # Rattling
            if include_rattling:
                self.collect_rattling(atoms, f"{struct_name}_rattle")

            # Strain
            if include_strain:
                self.collect_strain(atoms, f"{struct_name}_strain")

            # MD at multiple temperatures
            if include_md:
                self.collect_md_temperatures(atoms, f"{struct_name}_md")

            # Heating/cooling ramps
            if include_heating_cooling:
                self.collect_heating_ramp(atoms, f"{struct_name}_heat")
                self.collect_cooling_ramp(atoms, f"{struct_name}_cool")

            # NPT at various pressures
            if include_npt:
                self.collect_npt_pressures(atoms, f"{struct_name}_npt")

            # Subsystems
            if include_subsystems and metal_elements:
                subsystems = self.generate_subsystems(atoms, metal_elements)
                for sub_name, sub_atoms in subsystems:
                    full_name = f"{struct_name}_{sub_name}"

                    self.collect_single_point(sub_atoms, f"{full_name}_sp")

                    if include_opt:
                        self.collect_optimization(sub_atoms, f"{full_name}_opt")

                    if include_md:
                        # Fewer temperatures for subsystems
                        self.collect_md_temperatures(
                            sub_atoms, f"{full_name}_md",
                            temperatures=[300, 600, 1000]
                        )

        self.stats.collection_time = time.time() - start_time

        self._log(self.stats.summary())

        return self.stats

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about collected data."""
        info = {
            "db_path": self.db_path,
            "total_entries": len(self.db),
            "modes": {},
            "sources": set(),
            "formulas": set(),
        }

        for row in self.db.select():
            mode = row.get("mode", "unknown")
            info["modes"][mode] = info["modes"].get(mode, 0) + 1

            source = row.get("source")
            if source:
                info["sources"].add(source)

            info["formulas"].add(row.formula)

        info["sources"] = list(info["sources"])
        info["formulas"] = list(info["formulas"])

        return info

    def get_training_data(
        self,
        max_force: Optional[float] = None,
        modes: Optional[List[str]] = None,
        formula: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Query training data from database.

        Args:
            max_force: Filter by maximum force (eV/Å)
            modes: Filter by collection modes
            formula: Filter by chemical formula
            limit: Maximum number of entries

        Returns:
            List of dictionaries with structure data
        """
        data = []

        selection = []
        if formula:
            selection.append(f"formula={formula}")

        for row in self.db.select(",".join(selection) if selection else None):
            # Apply filters
            if max_force and row.get("max_force", 0) > max_force:
                continue

            if modes and row.get("mode") not in modes:
                continue

            entry = {
                "id": row.id,
                "atoms": row.toatoms(),
                "energy": row.get("total_energy"),
                "forces": row.data.get("forces") if row.data else None,
                "stress": row.data.get("stress") if row.data else None,
                "mode": row.get("mode"),
                "source": row.get("source"),
                "metadata": {k: row.get(k) for k in row.key_value_pairs.keys()},
            }
            data.append(entry)

            if limit and len(data) >= limit:
                break

        return data

    def export_to_xyz(
        self,
        output_file: str,
        max_force: Optional[float] = None,
        extended_xyz: bool = True,
    ) -> int:
        """
        Export database to extended XYZ format.

        Args:
            output_file: Output file path
            max_force: Filter by maximum force
            extended_xyz: Include energy/forces in XYZ

        Returns:
            Number of structures exported
        """
        data = self.get_training_data(max_force=max_force)

        structures = []
        for entry in data:
            atoms = entry["atoms"].copy()

            # Remove calculator to avoid key collision
            atoms.calc = None

            if extended_xyz:
                atoms.info["energy"] = entry["energy"]
                atoms.info["config_type"] = entry["mode"]
                atoms.info["source"] = entry.get("source", "")
                if entry["forces"] is not None:
                    atoms.arrays["forces"] = np.array(entry["forces"])

            structures.append(atoms)

        write(output_file, structures, format="extxyz")

        self._log(f"Exported {len(structures)} structures to {output_file}")
        return len(structures)


# =============================================================================
# Convenience functions
# =============================================================================

def collect_from_cif_directory(
    cif_dir: str,
    calculator: Calculator,
    db_path: str = "training_data.db",
    **kwargs,
) -> CollectionStats:
    """
    Collect training data from all CIF files in a directory.

    Args:
        cif_dir: Directory containing CIF files
        calculator: ASE calculator (MLIP)
        db_path: Database path
        **kwargs: Additional arguments for collect_all

    Returns:
        Collection statistics
    """
    from glob import glob

    cif_files = glob(os.path.join(cif_dir, "*.cif"))
    structures = [read(f) for f in cif_files]

    collector = MLIPCollector(calculator, db_path)
    return collector.collect_all(structures, **kwargs)


def collect_for_gulp_fitting(
    structures: List[Atoms],
    calculator: Calculator,
    db_path: str = "gulp_training_data.db",
    metal_elements: List[str] = ["Nb", "W"],
) -> CollectionStats:
    """
    Collect comprehensive data for GULP potential fitting.

    Includes all sampling modes optimized for classical potential fitting.
    """
    config = CollectionConfig(
        db_path=db_path,
        md_temperatures=[100, 300, 500, 800, 1000, 1200, 1500],
        md_steps_per_temp=300,
        md_save_interval=5,
        rattle_stdev=[0.02, 0.05, 0.1, 0.15, 0.2],
        rattle_n_configs=20,
        strain_magnitudes=[-0.05, -0.03, -0.01, 0.01, 0.03, 0.05],
    )

    collector = MLIPCollector(calculator, db_path, config)

    return collector.collect_all(
        structures,
        include_md=True,
        include_opt=True,
        include_rattling=True,
        include_strain=True,
        include_heating_cooling=True,
        include_npt=True,
        include_subsystems=True,
        metal_elements=metal_elements,
    )
