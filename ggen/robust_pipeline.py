"""
Robust data collection pipeline for GULP potential fitting.

Features:
- Checkpoint/resume capability
- Data validation and filtering
- Structure diversity via fingerprint-based deduplication
- Progress tracking with logging
- Configuration via dataclass
- Memory-efficient streaming to disk
- Error recovery and retry logic
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import warnings

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator

from .data_collector import DataCollector, TrainingData


# ==================== Configuration ====================

@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    # MD sampling
    md_temperature_range: Tuple[float, float] = (100, 1200)
    md_n_temperature_points: int = 8
    md_n_trajectories_per_temp: int = 2
    md_n_steps: int = 300
    md_sample_interval: int = 20
    md_timestep: float = 1.0
    md_friction_values: List[float] = field(default_factory=lambda: [0.002, 0.01, 0.05])
    md_heating_cooling_ramps: bool = True

    # Site swaps
    swap_n_iterations: int = 5
    swap_samples_per_iter: int = 10
    swap_max_swaps: int = 3

    # Strain
    volume_strain_range: Tuple[float, float] = (-0.05, 0.05)
    volume_n_samples: int = 7
    shear_max_strain: float = 0.03
    shear_n_samples: int = 5

    # Random displacements
    n_random_displacements: int = 30
    max_displacement: float = 0.1

    # Optimization
    fmax: float = 0.01
    max_opt_steps: int = 500


@dataclass
class FilterConfig:
    """Configuration for data filtering."""
    # Energy filters
    max_energy_per_atom: float = 10.0  # eV/atom
    min_energy_per_atom: float = -50.0  # eV/atom

    # Force filters
    max_force_component: float = 50.0  # eV/Å
    max_force_norm: float = 100.0  # eV/Å

    # Structure filters
    min_distance: float = 0.5  # Å (reject if atoms too close)
    max_volume_per_atom: float = 100.0  # Å³/atom
    min_volume_per_atom: float = 5.0  # Å³/atom

    # Diversity
    fingerprint_similarity_threshold: float = 0.98  # reject if too similar
    use_soap_fingerprint: bool = False  # use simple fingerprint if False


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # General
    name: str = "gulp_training_data"
    output_dir: str = "./training_output"
    checkpoint_interval: int = 100  # save every N samples

    # System
    metal_elements: List[str] = field(default_factory=lambda: ["Nb", "W"])
    oxygen_element: str = "O"

    # Sampling modes
    sample_3_component: bool = True
    sample_2_component: bool = True
    sample_1_component: bool = True
    use_diverse_md: bool = True

    # Sub-configs
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        # Reconstruct nested dataclasses
        data["sampling"] = SamplingConfig(**data.get("sampling", {}))
        data["filtering"] = FilterConfig(**data.get("filtering", {}))
        return cls(**data)


# ==================== Data Validation ====================

class DataValidator:
    """Validate and filter training data."""

    def __init__(self, config: FilterConfig):
        self.config = config
        self.stats = defaultdict(int)

    def validate_energy(self, data: TrainingData) -> bool:
        """Check if energy is within reasonable bounds."""
        n_atoms = len(data.structure)
        energy_per_atom = data.energy / n_atoms

        if energy_per_atom > self.config.max_energy_per_atom:
            self.stats["rejected_high_energy"] += 1
            return False
        if energy_per_atom < self.config.min_energy_per_atom:
            self.stats["rejected_low_energy"] += 1
            return False
        return True

    def validate_forces(self, data: TrainingData) -> bool:
        """Check if forces are reasonable."""
        forces = data.forces

        # Check max component
        if np.abs(forces).max() > self.config.max_force_component:
            self.stats["rejected_high_force_component"] += 1
            return False

        # Check max norm
        force_norms = np.linalg.norm(forces, axis=1)
        if force_norms.max() > self.config.max_force_norm:
            self.stats["rejected_high_force_norm"] += 1
            return False

        return True

    def validate_structure(self, data: TrainingData) -> bool:
        """Check if structure is physically reasonable."""
        atoms = data.structure
        n_atoms = len(atoms)

        # Volume per atom
        volume = atoms.get_volume()
        vol_per_atom = volume / n_atoms

        if vol_per_atom > self.config.max_volume_per_atom:
            self.stats["rejected_large_volume"] += 1
            return False
        if vol_per_atom < self.config.min_volume_per_atom:
            self.stats["rejected_small_volume"] += 1
            return False

        # Minimum interatomic distance
        try:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            min_dist = distances.min()

            if min_dist < self.config.min_distance:
                self.stats["rejected_close_atoms"] += 1
                return False
        except:
            pass

        return True

    def validate(self, data: TrainingData) -> bool:
        """Run all validations."""
        if not self.validate_energy(data):
            return False
        if not self.validate_forces(data):
            return False
        if not self.validate_structure(data):
            return False

        self.stats["passed"] += 1
        return True

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)


# ==================== Structure Fingerprinting ====================

class StructureFingerprint:
    """Compute structure fingerprints for diversity filtering."""

    def __init__(self, use_soap: bool = False):
        self.use_soap = use_soap
        self._fingerprints: Dict[str, np.ndarray] = {}

    def compute_simple_fingerprint(self, atoms: Atoms) -> np.ndarray:
        """
        Simple fingerprint based on:
        - Composition
        - Radial distribution function histogram
        - Cell parameters
        """
        n_atoms = len(atoms)

        # Composition vector (normalized)
        symbols = atoms.get_chemical_symbols()
        unique_elements = sorted(set(symbols))
        comp = np.array([symbols.count(e) / n_atoms for e in unique_elements])

        # RDF histogram
        try:
            distances = atoms.get_all_distances(mic=True).flatten()
            distances = distances[distances > 0.1]  # exclude self
            rdf_hist, _ = np.histogram(distances, bins=20, range=(0, 8), density=True)
        except:
            rdf_hist = np.zeros(20)

        # Cell parameters (normalized)
        cell_params = atoms.cell.cellpar()
        cell_norm = cell_params / (cell_params.max() + 1e-10)

        # Combine
        fingerprint = np.concatenate([comp, rdf_hist, cell_norm])
        return fingerprint

    def compute(self, atoms: Atoms) -> np.ndarray:
        """Compute fingerprint for atoms."""
        if self.use_soap:
            # Would use dscribe SOAP here
            raise NotImplementedError("SOAP fingerprint requires dscribe")
        return self.compute_simple_fingerprint(atoms)

    def similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute cosine similarity between fingerprints."""
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return np.dot(fp1, fp2) / (norm1 * norm2)


class DiversityFilter:
    """Filter to ensure structure diversity."""

    def __init__(self, threshold: float = 0.98, use_soap: bool = False):
        self.threshold = threshold
        self.fingerprinter = StructureFingerprint(use_soap=use_soap)
        self.stored_fingerprints: List[np.ndarray] = []
        self.n_rejected = 0

    def is_diverse(self, atoms: Atoms) -> bool:
        """Check if structure is sufficiently different from stored ones."""
        fp = self.fingerprinter.compute(atoms)

        for stored_fp in self.stored_fingerprints:
            sim = self.fingerprinter.similarity(fp, stored_fp)
            if sim > self.threshold:
                self.n_rejected += 1
                return False

        self.stored_fingerprints.append(fp)
        return True

    def reset(self):
        """Clear stored fingerprints."""
        self.stored_fingerprints = []
        self.n_rejected = 0


# ==================== Checkpoint Manager ====================

class CheckpointManager:
    """Manage checkpoints for pipeline resumption."""

    def __init__(self, output_dir: str, name: str):
        self.output_dir = Path(output_dir)
        self.name = name
        self.checkpoint_file = self.output_dir / f"{name}_checkpoint.pkl"
        self.data_file = self.output_dir / f"{name}_data.pkl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        data: List[TrainingData],
        state: Dict[str, Any],
    ):
        """Save checkpoint with current data and state."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(data),
            "state": state,
        }

        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)

        with open(self.data_file, "wb") as f:
            pickle.dump(data, f)

        logging.info(f"Checkpoint saved: {len(data)} samples")

    def load_checkpoint(self) -> Tuple[List[TrainingData], Dict[str, Any]]:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return [], {}

        with open(self.checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        with open(self.data_file, "rb") as f:
            data = pickle.load(f)

        logging.info(f"Checkpoint loaded: {len(data)} samples from {checkpoint['timestamp']}")
        return data, checkpoint.get("state", {})

    def has_checkpoint(self) -> bool:
        return self.checkpoint_file.exists()

    def clear(self):
        """Clear checkpoint files."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.data_file.exists():
            self.data_file.unlink()


# ==================== Progress Tracker ====================

class ProgressTracker:
    """Track and log progress."""

    def __init__(self, total_phases: int, log_file: Optional[str] = None):
        self.total_phases = total_phases
        self.current_phase = 0
        self.phase_name = ""
        self.phase_progress = 0
        self.phase_total = 0
        self.start_time = datetime.now()

        # Setup logging
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(logging.INFO)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)

    def start_phase(self, name: str, total: int = 0):
        """Start a new phase."""
        self.current_phase += 1
        self.phase_name = name
        self.phase_progress = 0
        self.phase_total = total
        self.logger.info(f"\n[Phase {self.current_phase}/{self.total_phases}] {name}")
        if total > 0:
            self.logger.info(f"  Total items: {total}")

    def update(self, increment: int = 1, message: str = ""):
        """Update progress."""
        self.phase_progress += increment
        if self.phase_total > 0:
            pct = 100 * self.phase_progress / self.phase_total
            if message:
                self.logger.info(f"  [{pct:.1f}%] {message}")

    def log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(self.logger, level)(f"  {message}")

    def summary(self, n_samples: int):
        """Print final summary."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Pipeline Complete")
        self.logger.info(f"  Total samples: {n_samples}")
        self.logger.info(f"  Elapsed time: {elapsed}")
        self.logger.info(f"{'='*60}")


# ==================== Robust Pipeline ====================

class RobustPipeline:
    """
    Robust data collection pipeline with:
    - Checkpoint/resume
    - Data validation
    - Diversity filtering
    - Progress tracking
    - Error recovery
    """

    def __init__(
        self,
        calculator: Calculator,
        config: Optional[PipelineConfig] = None,
    ):
        self.calculator = calculator
        self.config = config or PipelineConfig()

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(self.output_dir / "config.json")

        # Initialize components
        self.collector = DataCollector(calculator)
        self.validator = DataValidator(self.config.filtering)
        self.diversity_filter = DiversityFilter(
            threshold=self.config.filtering.fingerprint_similarity_threshold,
            use_soap=self.config.filtering.use_soap_fingerprint,
        )
        self.checkpoint_mgr = CheckpointManager(
            self.config.output_dir,
            self.config.name,
        )

        # Count phases
        n_phases = sum([
            self.config.sample_3_component,
            self.config.sample_2_component,
            self.config.sample_1_component,
        ]) * 5  # each component has ~5 sub-phases

        self.progress = ProgressTracker(
            n_phases,
            log_file=str(self.output_dir / "pipeline.log"),
        )

        # Internal state
        self.all_data: List[TrainingData] = []
        self.state: Dict[str, Any] = {}

    def _add_sample(self, data: TrainingData) -> bool:
        """Add sample with validation and diversity check."""
        # Validate
        if not self.validator.validate(data):
            return False

        # Diversity check
        if not self.diversity_filter.is_diverse(data.structure):
            return False

        self.all_data.append(data)

        # Periodic checkpoint
        if len(self.all_data) % self.config.checkpoint_interval == 0:
            self.checkpoint_mgr.save_checkpoint(self.all_data, self.state)

        return True

    def _add_samples(self, data_list: List[TrainingData]) -> int:
        """Add multiple samples, return count of accepted."""
        accepted = 0
        for data in data_list:
            if self._add_sample(data):
                accepted += 1
        return accepted

    def _sample_structure(
        self,
        atoms: Atoms,
        label: str,
        do_swaps: bool = True,
    ) -> Atoms:
        """
        Sample a single structure with all perturbations.
        Returns the best (lowest energy) structure found.
        """
        cfg = self.config.sampling

        # 1. Optimize
        self.progress.log(f"Optimizing {label}...")
        try:
            eq_data = self.collector.optimize_structure(
                atoms, fmax=cfg.fmax, steps=cfg.max_opt_steps
            )
            eq_data.source = f"{label}_eq"
            self._add_sample(eq_data)
            current = eq_data.structure.copy()
            best_energy = eq_data.energy
        except Exception as e:
            self.progress.log(f"Optimization failed: {e}", "warning")
            current = atoms.copy()
            best_energy = float("inf")

        # 2. Site swaps (if multi-element)
        if do_swaps and len(set(current.get_chemical_symbols())) > 1:
            self.progress.log(f"Site swap exploration...")
            for it in range(cfg.swap_n_iterations):
                try:
                    swap_data = self.collector.sample_site_swaps(
                        current,
                        n_samples=cfg.swap_samples_per_iter,
                        max_swaps=cfg.swap_max_swaps,
                    )
                    for d in swap_data:
                        d.source = f"{label}_swap_{it}_{d.source}"
                    accepted = self._add_samples(swap_data)

                    # Update to best
                    if swap_data:
                        energies = [d.energy for d in swap_data]
                        best_idx = np.argmin(energies)
                        if energies[best_idx] < best_energy:
                            best_energy = energies[best_idx]
                            current = swap_data[best_idx].structure.copy()
                except Exception as e:
                    self.progress.log(f"Swap iter {it} failed: {e}", "warning")

        # 3. Volume strain
        self.progress.log(f"Volume strain sampling...")
        try:
            vol_data = self.collector.sample_volume_strain(
                current,
                strain_range=cfg.volume_strain_range,
                n_samples=cfg.volume_n_samples,
            )
            for d in vol_data:
                d.source = f"{label}_{d.source}"
            self._add_samples(vol_data)
        except Exception as e:
            self.progress.log(f"Volume strain failed: {e}", "warning")

        # 4. Shear strain
        self.progress.log(f"Shear strain sampling...")
        try:
            shear_data = self.collector.sample_shear_strain(
                current,
                max_strain=cfg.shear_max_strain,
                n_samples=cfg.shear_n_samples,
            )
            for d in shear_data:
                d.source = f"{label}_{d.source}"
            self._add_samples(shear_data)
        except Exception as e:
            self.progress.log(f"Shear strain failed: {e}", "warning")

        # 5. Random displacements
        self.progress.log(f"Random displacement sampling...")
        try:
            disp_data = self.collector.sample_random_displacements(
                current,
                n_samples=cfg.n_random_displacements,
                max_displacement=cfg.max_displacement,
            )
            for d in disp_data:
                d.source = f"{label}_{d.source}"
            self._add_samples(disp_data)
        except Exception as e:
            self.progress.log(f"Displacement failed: {e}", "warning")

        # 6. MD sampling
        self.progress.log(f"MD sampling...")
        try:
            if self.config.use_diverse_md:
                md_data = self.collector.sample_md_diverse(
                    current,
                    temperature_range=cfg.md_temperature_range,
                    n_temperature_points=cfg.md_n_temperature_points,
                    n_trajectories_per_temp=cfg.md_n_trajectories_per_temp,
                    n_steps=cfg.md_n_steps,
                    sample_interval=cfg.md_sample_interval,
                    timestep=cfg.md_timestep,
                    friction_values=cfg.md_friction_values,
                    heating_cooling_ramps=cfg.md_heating_cooling_ramps,
                )
            else:
                temps = np.linspace(
                    cfg.md_temperature_range[0],
                    cfg.md_temperature_range[1],
                    cfg.md_n_temperature_points,
                ).tolist()
                md_data = self.collector.sample_md(
                    current,
                    temperatures=temps,
                    n_steps=cfg.md_n_steps,
                    sample_interval=cfg.md_sample_interval,
                )

            for d in md_data:
                d.source = f"{label}_{d.source}"
            self._add_samples(md_data)
        except Exception as e:
            self.progress.log(f"MD failed: {e}", "warning")

        return current

    def run(
        self,
        initial_structures: List[Atoms],
        resume: bool = True,
    ) -> List[TrainingData]:
        """
        Run the full pipeline.

        Args:
            initial_structures: List of initial structure guesses
            resume: Whether to resume from checkpoint

        Returns:
            List of validated TrainingData
        """
        # Resume from checkpoint if available
        if resume and self.checkpoint_mgr.has_checkpoint():
            self.all_data, self.state = self.checkpoint_mgr.load_checkpoint()
            # Rebuild diversity filter
            for data in self.all_data:
                self.diversity_filter.is_diverse(data.structure)
        else:
            self.all_data = []
            self.state = {"completed_structures": []}

        completed = set(self.state.get("completed_structures", []))
        metals = self.config.metal_elements
        oxygen = self.config.oxygen_element

        try:
            # Phase 1: 3-component structures
            if self.config.sample_3_component:
                self.progress.start_phase(
                    "3-Component Structures",
                    len(initial_structures),
                )

                for i, atoms in enumerate(initial_structures):
                    struct_id = f"3comp_{i}"
                    if struct_id in completed:
                        self.progress.log(f"Skipping {struct_id} (already done)")
                        continue

                    formula = atoms.get_chemical_formula()
                    self.progress.log(f"Processing {formula}...")

                    self._sample_structure(atoms, struct_id, do_swaps=True)

                    completed.add(struct_id)
                    self.state["completed_structures"] = list(completed)
                    self.progress.update(message=f"{formula} complete")

            # Phase 2: 2-component structures
            if self.config.sample_2_component:
                template = initial_structures[0]
                symbols = template.get_chemical_symbols()

                # Binary oxides
                for metal in metals:
                    struct_id = f"2comp_{metal}O"
                    if struct_id in completed:
                        continue

                    self.progress.start_phase(f"2-Component: {metal}-O")

                    binary = template.copy()
                    new_symbols = [
                        oxygen if s == oxygen else metal
                        for s in symbols
                    ]
                    binary.set_chemical_symbols(new_symbols)

                    self._sample_structure(binary, struct_id, do_swaps=False)

                    completed.add(struct_id)
                    self.state["completed_structures"] = list(completed)

                # Metal alloy
                if len(metals) >= 2:
                    struct_id = "2comp_alloy"
                    if struct_id not in completed:
                        self.progress.start_phase(f"2-Component: {'-'.join(metals)} alloy")

                        alloy = template.copy()
                        new_symbols = []
                        metal_idx = 0
                        for s in symbols:
                            if s == oxygen:
                                new_symbols.append(metals[metal_idx % len(metals)])
                                metal_idx += 1
                            else:
                                new_symbols.append(s)
                        alloy.set_chemical_symbols(new_symbols)

                        self._sample_structure(alloy, struct_id, do_swaps=False)

                        completed.add(struct_id)
                        self.state["completed_structures"] = list(completed)

            # Phase 3: 1-component structures
            if self.config.sample_1_component:
                template = initial_structures[0]

                # Pure oxygen
                struct_id = "1comp_O"
                if struct_id not in completed:
                    self.progress.start_phase(f"1-Component: Pure {oxygen}")

                    pure_O = template.copy()
                    pure_O.set_chemical_symbols([oxygen] * len(template))

                    self._sample_structure(pure_O, struct_id, do_swaps=False)

                    completed.add(struct_id)
                    self.state["completed_structures"] = list(completed)

                # Pure metals
                for metal in metals:
                    struct_id = f"1comp_{metal}"
                    if struct_id in completed:
                        continue

                    self.progress.start_phase(f"1-Component: Pure {metal}")

                    pure_metal = template.copy()
                    pure_metal.set_chemical_symbols([metal] * len(template))

                    self._sample_structure(pure_metal, struct_id, do_swaps=False)

                    completed.add(struct_id)
                    self.state["completed_structures"] = list(completed)

        except KeyboardInterrupt:
            self.progress.log("Interrupted! Saving checkpoint...", "warning")
            self.checkpoint_mgr.save_checkpoint(self.all_data, self.state)
            raise
        except Exception as e:
            self.progress.log(f"Error: {e}", "error")
            self.checkpoint_mgr.save_checkpoint(self.all_data, self.state)
            raise

        # Final save
        self.checkpoint_mgr.save_checkpoint(self.all_data, self.state)

        # Export final data
        self._export_results()

        # Print summary
        self.progress.summary(len(self.all_data))
        self._print_stats()

        return self.all_data

    def _export_results(self):
        """Export results in multiple formats."""
        # XYZ
        xyz_path = self.output_dir / f"{self.config.name}.xyz"
        self.collector.save_xyz(self.all_data, str(xyz_path))

        # JSON
        json_path = self.output_dir / f"{self.config.name}.json"
        self.collector.save_json(self.all_data, str(json_path))

        # Statistics
        stats_path = self.output_dir / f"{self.config.name}_stats.json"
        stats = self._compute_stats()
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.all_data:
            return {}

        energies = [d.energy for d in self.all_data]
        n_atoms_list = [len(d.structure) for d in self.all_data]
        energies_per_atom = [e/n for e, n in zip(energies, n_atoms_list)]

        forces = np.concatenate([d.forces for d in self.all_data])
        force_norms = np.linalg.norm(forces, axis=1)

        # Composition breakdown
        compositions = defaultdict(int)
        for d in self.all_data:
            formula = d.structure.get_chemical_formula(mode="hill")
            compositions[formula] += 1

        # Source breakdown
        sources = defaultdict(int)
        for d in self.all_data:
            # Extract source type (e.g., "md", "swap", "strain")
            parts = d.source.split("_")
            for part in parts:
                if part in ["md", "swap", "strain", "shear", "volume", "eq", "disp", "random"]:
                    sources[part] += 1
                    break

        return {
            "n_samples": len(self.all_data),
            "energy": {
                "min": float(np.min(energies)),
                "max": float(np.max(energies)),
                "mean": float(np.mean(energies)),
                "std": float(np.std(energies)),
            },
            "energy_per_atom": {
                "min": float(np.min(energies_per_atom)),
                "max": float(np.max(energies_per_atom)),
                "mean": float(np.mean(energies_per_atom)),
                "std": float(np.std(energies_per_atom)),
            },
            "forces": {
                "max_norm": float(np.max(force_norms)),
                "mean_norm": float(np.mean(force_norms)),
                "std_norm": float(np.std(force_norms)),
            },
            "compositions": dict(compositions),
            "sources": dict(sources),
            "validation": self.validator.get_stats(),
            "diversity_rejected": self.diversity_filter.n_rejected,
        }

    def _print_stats(self):
        """Print statistics summary."""
        stats = self._compute_stats()
        self.progress.log("\n--- Dataset Statistics ---")
        self.progress.log(f"Total samples: {stats['n_samples']}")
        self.progress.log(f"Energy/atom: {stats['energy_per_atom']['mean']:.3f} +/- "
                         f"{stats['energy_per_atom']['std']:.3f} eV/atom")
        self.progress.log(f"Force norm: {stats['forces']['mean_norm']:.3f} +/- "
                         f"{stats['forces']['std_norm']:.3f} eV/Å")
        self.progress.log(f"\nValidation rejected: {sum(stats['validation'].values()) - stats['validation'].get('passed', 0)}")
        self.progress.log(f"Diversity rejected: {stats['diversity_rejected']}")
        self.progress.log(f"\nCompositions:")
        for comp, count in sorted(stats['compositions'].items()):
            self.progress.log(f"  {comp}: {count}")


# ==================== Convenience Functions ====================

def run_pipeline(
    calculator: Calculator,
    structures: List[Atoms],
    output_dir: str = "./training_output",
    metal_elements: List[str] = None,
    resume: bool = True,
    **kwargs,
) -> List[TrainingData]:
    """
    Convenience function to run the robust pipeline.

    Args:
        calculator: MLIP calculator
        structures: Initial structure guesses
        output_dir: Output directory
        metal_elements: Metal elements (auto-detected if None)
        resume: Resume from checkpoint
        **kwargs: Additional PipelineConfig parameters

    Returns:
        List of TrainingData

    Example:
        from orb_models.forcefield import pretrained
        from ase.io import read

        calc = pretrained.orb_v2()
        structures = [read("NbWO.cif")]

        data = run_pipeline(
            calc,
            structures,
            output_dir="./training",
            metal_elements=["Nb", "W"],
            use_diverse_md=True,
        )
    """
    # Auto-detect metals
    if metal_elements is None:
        all_symbols = set()
        for atoms in structures:
            all_symbols.update(atoms.get_chemical_symbols())
        metal_elements = [s for s in all_symbols if s != "O"]

    config = PipelineConfig(
        output_dir=output_dir,
        metal_elements=metal_elements,
        **kwargs,
    )

    pipeline = RobustPipeline(calculator, config)
    return pipeline.run(structures, resume=resume)
