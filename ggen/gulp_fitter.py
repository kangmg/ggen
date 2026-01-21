"""
GULP Potential Fitter with Shell Model and QEq support.

Integrates ParamGULP functionality with extensions for:
- Buckingham + Shell Model (polarizable ions)
- Buckingham + QEq (charge equilibration)

Based on ParamGULP by José Diogo L. Dutra et al.
Original: https://github.com/chejunwei2/ParamGULP
License: MIT
"""

import os
import re
import tempfile
import shutil
import subprocess
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import json

import numpy as np
from scipy.optimize import dual_annealing, minimize, OptimizeResult

from ase import Atoms
from ase.io import read as ase_read


class ChargeModel(Enum):
    """Charge model for GULP potential."""
    FIXED = "fixed"       # Fixed formal charges
    SHELL = "shell"       # Core-shell model (polarizable)
    QEQ = "qeq"           # Charge equilibration
    EEM = "eem"           # Electronegativity equalization


class PotentialType(Enum):
    """Interatomic potential type."""
    BUCKINGHAM = "buckingham"
    LENNARD_JONES = "lennard"
    MORSE = "morse"


@dataclass
class BuckinghamParams:
    """Buckingham potential parameters: V(r) = A*exp(-r/rho) - C/r^6"""
    A: float               # Pre-exponential factor (eV)
    rho: float             # Length scale (Å)
    C: float = 0.0         # Dispersion coefficient (eV·Å^6)
    rmin: float = 0.0      # Minimum cutoff (Å)
    rmax: float = 12.0     # Maximum cutoff (Å)

    # Bounds for fitting
    A_bounds: Tuple[float, float] = (100.0, 50000.0)
    rho_bounds: Tuple[float, float] = (0.1, 1.0)
    C_bounds: Tuple[float, float] = (0.0, 100.0)


@dataclass
class ShellParams:
    """Shell model parameters for polarizable ions."""
    element: str
    core_charge: float     # Core charge
    shell_charge: float    # Shell charge (usually negative for anions)
    spring_k: float        # Spring constant (eV/Å^2)

    # Bounds for fitting
    spring_k_bounds: Tuple[float, float] = (1.0, 500.0)

    @property
    def total_charge(self) -> float:
        return self.core_charge + self.shell_charge


@dataclass
class QEqParams:
    """QEq parameters for charge equilibration."""
    element: str
    chi: float             # Electronegativity (eV)
    mu: float              # Chemical hardness / self-Coulomb (eV)
    q_min: float = -3.0    # Minimum charge
    q_max: float = 3.0     # Maximum charge

    # Bounds for fitting
    chi_bounds: Tuple[float, float] = (0.0, 15.0)
    mu_bounds: Tuple[float, float] = (1.0, 20.0)


@dataclass
class EEMParams:
    """EEM (Electronegativity Equalization Method) parameters."""
    element: str
    chi: float             # Electronegativity
    mu: float              # Hardness
    r_eem: float = 0.0     # EEM radius (optional)

    chi_bounds: Tuple[float, float] = (0.0, 15.0)
    mu_bounds: Tuple[float, float] = (1.0, 20.0)


@dataclass
class PotentialConfig:
    """Complete potential configuration."""
    name: str = "NbWO_potential"
    charge_model: ChargeModel = ChargeModel.SHELL
    potential_type: PotentialType = PotentialType.BUCKINGHAM

    # Species and their parameters
    buckingham: Dict[Tuple[str, str], BuckinghamParams] = field(default_factory=dict)
    shell_params: Dict[str, ShellParams] = field(default_factory=dict)
    qeq_params: Dict[str, QEqParams] = field(default_factory=dict)
    eem_params: Dict[str, EEMParams] = field(default_factory=dict)

    # Fixed species charges (for FIXED charge model)
    fixed_charges: Dict[str, float] = field(default_factory=dict)

    # Fitting configuration
    fit_buckingham: bool = True
    fit_charges: bool = False  # Fit shell/QEq parameters

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "charge_model": self.charge_model.value,
            "potential_type": self.potential_type.value,
            "buckingham": {
                f"{k[0]}-{k[1]}": {
                    "A": v.A, "rho": v.rho, "C": v.C,
                    "rmin": v.rmin, "rmax": v.rmax,
                    "A_bounds": v.A_bounds, "rho_bounds": v.rho_bounds, "C_bounds": v.C_bounds
                }
                for k, v in self.buckingham.items()
            },
            "shell_params": {
                k: {"core_charge": v.core_charge, "shell_charge": v.shell_charge,
                    "spring_k": v.spring_k, "spring_k_bounds": v.spring_k_bounds}
                for k, v in self.shell_params.items()
            },
            "qeq_params": {
                k: {"chi": v.chi, "mu": v.mu, "q_min": v.q_min, "q_max": v.q_max,
                    "chi_bounds": v.chi_bounds, "mu_bounds": v.mu_bounds}
                for k, v in self.qeq_params.items()
            },
            "fixed_charges": self.fixed_charges,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PotentialConfig":
        """Create from dictionary."""
        config = cls(
            name=d.get("name", "potential"),
            charge_model=ChargeModel(d.get("charge_model", "shell")),
            potential_type=PotentialType(d.get("potential_type", "buckingham")),
        )

        # Parse Buckingham params
        for pair_str, params in d.get("buckingham", {}).items():
            elem1, elem2 = pair_str.split("-")
            config.buckingham[(elem1, elem2)] = BuckinghamParams(
                A=params["A"], rho=params["rho"], C=params.get("C", 0.0),
                rmin=params.get("rmin", 0.0), rmax=params.get("rmax", 12.0),
                A_bounds=tuple(params.get("A_bounds", (100, 50000))),
                rho_bounds=tuple(params.get("rho_bounds", (0.1, 1.0))),
                C_bounds=tuple(params.get("C_bounds", (0, 100))),
            )

        # Parse shell params
        for elem, params in d.get("shell_params", {}).items():
            config.shell_params[elem] = ShellParams(
                element=elem,
                core_charge=params["core_charge"],
                shell_charge=params["shell_charge"],
                spring_k=params["spring_k"],
                spring_k_bounds=tuple(params.get("spring_k_bounds", (1, 500))),
            )

        # Parse QEq params
        for elem, params in d.get("qeq_params", {}).items():
            config.qeq_params[elem] = QEqParams(
                element=elem,
                chi=params["chi"], mu=params["mu"],
                q_min=params.get("q_min", -3), q_max=params.get("q_max", 3),
                chi_bounds=tuple(params.get("chi_bounds", (0, 15))),
                mu_bounds=tuple(params.get("mu_bounds", (1, 20))),
            )

        config.fixed_charges = d.get("fixed_charges", {})
        return config


@dataclass
class FitTarget:
    """Target data for fitting."""
    name: str                      # Structure identifier
    atoms: Atoms                   # ASE Atoms object
    energy: Optional[float] = None # Target energy (eV)
    forces: Optional[np.ndarray] = None  # Target forces
    stress: Optional[np.ndarray] = None  # Target stress
    lattice_params: Optional[Dict[str, float]] = None  # a, b, c, alpha, beta, gamma

    # Weights for objective function
    energy_weight: float = 1.0
    force_weight: float = 0.1
    lattice_weight: float = 100.0


@dataclass
class FitResult:
    """Result of potential fitting."""
    config: PotentialConfig
    objective_value: float
    n_iterations: int
    converged: bool
    history: List[float] = field(default_factory=list)
    message: str = ""


class GULPFitter:
    """
    GULP Potential Fitter with Shell Model and QEq support.

    Supports:
    - Buckingham + Fixed charges
    - Buckingham + Shell Model (core-shell with spring)
    - Buckingham + QEq (charge equilibration)
    - Buckingham + EEM (electronegativity equalization)

    Example:
        # Create fitter
        fitter = GULPFitter(gulp_command="/path/to/gulp")

        # Define potential configuration
        config = PotentialConfig(
            charge_model=ChargeModel.SHELL,
            buckingham={
                ("Nb", "O"): BuckinghamParams(A=1000, rho=0.35),
                ("W", "O"): BuckinghamParams(A=1200, rho=0.34),
                ("O", "O"): BuckinghamParams(A=22764, rho=0.149, C=27.88),
            },
            shell_params={
                "O": ShellParams("O", core_charge=0.9, shell_charge=-2.9, spring_k=74.92),
            },
            fixed_charges={"Nb": 5.0, "W": 6.0},
        )

        # Add training structures
        targets = [FitTarget(name="NbWO4", atoms=atoms, energy=-100.0, ...)]

        # Run fitting
        result = fitter.fit(config, targets, method="dual_annealing")
    """

    MAX_OBJECTIVE_VALUE = 1e10

    def __init__(
        self,
        gulp_command: Optional[str] = None,
        gulp_lib: Optional[str] = None,
        working_dir: Optional[str] = None,
        timeout: int = 120,
        verbose: bool = True,
    ):
        """
        Initialize GULP fitter.

        Args:
            gulp_command: Path to GULP executable
            gulp_lib: Path to GULP library directory
            working_dir: Working directory for calculations
            timeout: Timeout for GULP calculations (seconds)
            verbose: Print progress information
        """
        self.gulp_command = gulp_command or os.environ.get("GULP_EXE", "gulp")
        self.gulp_lib = gulp_lib or os.environ.get("GULP_LIB", "")
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="gulp_fit_")
        self.timeout = timeout
        self.verbose = verbose

        # Fitting state
        self._iteration = 0
        self._best_objective = self.MAX_OBJECTIVE_VALUE
        self._history = []
        self._cache = {}

        os.makedirs(self.working_dir, exist_ok=True)

    def _log(self, msg: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            print(msg)

    def generate_species_block(
        self,
        config: PotentialConfig,
        elements: List[str],
    ) -> str:
        """Generate GULP species block based on charge model."""
        lines = ["species"]

        if config.charge_model == ChargeModel.SHELL:
            for elem in elements:
                if elem in config.shell_params:
                    sp = config.shell_params[elem]
                    lines.append(f"{elem} core {sp.core_charge:.4f}")
                    lines.append(f"{elem} shell {sp.shell_charge:.4f}")
                elif elem in config.fixed_charges:
                    lines.append(f"{elem} core {config.fixed_charges[elem]:.4f}")
                else:
                    # Default: assume formal oxidation state
                    lines.append(f"{elem} core 0.0")

        elif config.charge_model == ChargeModel.QEQ:
            # QEq uses initial charges that get equilibrated
            for elem in elements:
                if elem in config.fixed_charges:
                    lines.append(f"{elem} core {config.fixed_charges[elem]:.4f}")
                else:
                    lines.append(f"{elem} core 0.0")

        elif config.charge_model == ChargeModel.EEM:
            for elem in elements:
                if elem in config.fixed_charges:
                    lines.append(f"{elem} core {config.fixed_charges[elem]:.4f}")
                else:
                    lines.append(f"{elem} core 0.0")

        else:  # FIXED
            for elem in elements:
                charge = config.fixed_charges.get(elem, 0.0)
                lines.append(f"{elem} core {charge:.4f}")

        return "\n".join(lines)

    def generate_buckingham_block(
        self,
        config: PotentialConfig,
        with_bounds: bool = False,
    ) -> str:
        """Generate Buckingham potential block."""
        lines = ["buckingham"]

        for (elem1, elem2), params in config.buckingham.items():
            # Determine species types based on charge model
            if config.charge_model == ChargeModel.SHELL:
                type1 = "shel" if elem1 in config.shell_params else "core"
                type2 = "shel" if elem2 in config.shell_params else "core"
            else:
                type1 = type2 = "core"

            if with_bounds:
                # ParamGULP style: value_lower_upper
                A_str = f"{params.A:.4f}_{params.A_bounds[0]}_{params.A_bounds[1]}"
                rho_str = f"{params.rho:.4f}_{params.rho_bounds[0]}_{params.rho_bounds[1]}"
                C_str = f"{params.C:.4f}"  # Usually not fitted
                lines.append(
                    f"{elem1} {type1} {elem2} {type2} {A_str} {rho_str} {C_str} "
                    f"{params.rmin} {params.rmax}"
                )
            else:
                lines.append(
                    f"{elem1} {type1} {elem2} {type2} {params.A:.4f} {params.rho:.6f} "
                    f"{params.C:.4f} {params.rmin} {params.rmax}"
                )

        return "\n".join(lines)

    def generate_spring_block(self, config: PotentialConfig) -> str:
        """Generate spring constants for shell model."""
        if config.charge_model != ChargeModel.SHELL:
            return ""

        lines = ["spring"]
        for elem, sp in config.shell_params.items():
            lines.append(f"{elem} {sp.spring_k:.4f}")

        return "\n".join(lines)

    def generate_qeq_block(self, config: PotentialConfig) -> str:
        """Generate QEq parameters block."""
        if config.charge_model != ChargeModel.QEQ:
            return ""

        lines = ["qeq"]
        for elem, qp in config.qeq_params.items():
            lines.append(f"{elem} {qp.chi:.6f} {qp.mu:.6f} {qp.q_min:.2f} {qp.q_max:.2f}")

        return "\n".join(lines)

    def generate_eem_block(self, config: PotentialConfig) -> str:
        """Generate EEM parameters block."""
        if config.charge_model != ChargeModel.EEM:
            return ""

        lines = ["eem"]
        for elem, ep in config.eem_params.items():
            if ep.r_eem > 0:
                lines.append(f"{elem} {ep.chi:.6f} {ep.mu:.6f} {ep.r_eem:.4f}")
            else:
                lines.append(f"{elem} {ep.chi:.6f} {ep.mu:.6f}")

        return "\n".join(lines)

    def generate_structure_block(
        self,
        target: FitTarget,
        config: PotentialConfig,
    ) -> str:
        """Generate structure definition block for GULP."""
        atoms = target.atoms
        cell = atoms.get_cell()
        positions = atoms.get_scaled_positions()
        symbols = atoms.get_chemical_symbols()

        # Get cell parameters
        a, b, c = cell.lengths()
        alpha, beta, gamma = cell.angles()

        lines = [
            f"name {target.name}",
            "cell",
            f"{a:.6f} {b:.6f} {c:.6f} {alpha:.4f} {beta:.4f} {gamma:.4f}",
            "fractional",
        ]

        for i, (sym, pos) in enumerate(zip(symbols, positions)):
            if config.charge_model == ChargeModel.SHELL and sym in config.shell_params:
                # Add both core and shell at same position
                lines.append(f"{sym} core {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
                lines.append(f"{sym} shel {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
            else:
                lines.append(f"{sym} core {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

        return "\n".join(lines)

    def generate_observables_block(self, target: FitTarget) -> str:
        """Generate observables block for fitting."""
        lines = ["observables"]

        if target.energy is not None:
            lines.append("energy eV")
            lines.append(f"  {target.energy:.6f} {target.energy_weight:.4f}")

        # Lattice parameters from atoms
        if target.lattice_params:
            lines.append("cell")
            for param, value in target.lattice_params.items():
                lines.append(f"  {param} {value:.6f}")

        lines.append("end")
        return "\n".join(lines)

    def generate_gulp_input(
        self,
        config: PotentialConfig,
        targets: List[FitTarget],
        keywords: str = "conp optimise compare",
        for_fitting: bool = False,
    ) -> str:
        """
        Generate complete GULP input file.

        Args:
            config: Potential configuration
            targets: List of target structures
            keywords: GULP keywords
            for_fitting: If True, include observables for fitting

        Returns:
            Complete GULP input file content
        """
        # Get all unique elements
        all_elements = set()
        for target in targets:
            all_elements.update(target.atoms.get_chemical_symbols())
        all_elements = sorted(all_elements)

        # Add QEq/EEM keyword if needed
        if config.charge_model == ChargeModel.QEQ:
            if "qeq" not in keywords.lower():
                keywords = "qeq " + keywords
        elif config.charge_model == ChargeModel.EEM:
            if "eem" not in keywords.lower():
                keywords = "eem " + keywords

        sections = [keywords]

        # Add each structure
        for target in targets:
            sections.append(self.generate_structure_block(target, config))
            if for_fitting:
                sections.append(self.generate_observables_block(target))
            sections.append("")  # Empty line between structures

        # Species block
        sections.append(self.generate_species_block(config, all_elements))

        # Potential blocks
        sections.append(self.generate_buckingham_block(config))

        # Charge model specific blocks
        if config.charge_model == ChargeModel.SHELL:
            sections.append(self.generate_spring_block(config))
        elif config.charge_model == ChargeModel.QEQ:
            sections.append(self.generate_qeq_block(config))
        elif config.charge_model == ChargeModel.EEM:
            sections.append(self.generate_eem_block(config))

        return "\n\n".join(sections)

    def run_gulp(
        self,
        input_content: str,
        label: str = "gulp",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Run GULP calculation.

        Returns:
            (success, output_content, parsed_results)
        """
        gin_file = os.path.join(self.working_dir, f"{label}.gin")
        gout_file = os.path.join(self.working_dir, f"{label}.gout")

        with open(gin_file, "w") as f:
            f.write(input_content)

        # Set environment
        env = os.environ.copy()
        if self.gulp_lib:
            env["GULP_LIB"] = self.gulp_lib

        # Run GULP
        try:
            result = subprocess.run(
                [self.gulp_command, label, label],
                cwd=self.working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "Timeout", {}
        except Exception as e:
            return False, str(e), {}

        # Read output
        if not os.path.exists(gout_file):
            return False, "No output file", {}

        with open(gout_file, "r") as f:
            output = f.read()

        # Check for completion
        if "Job Finished" not in output:
            return False, output, {}

        # Parse results
        results = self._parse_gulp_output(output)

        return True, output, results

    def _parse_gulp_output(self, output: str) -> Dict[str, Any]:
        """Parse GULP output file for results."""
        results = {
            "structures": {},
            "energies": {},
            "final_cells": {},
            "final_gnorm": None,
        }

        current_name = None
        lines = output.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Structure name - formats vary:
            # "**** Output for configuration   1 ****"
            # "Output for configuration   1 : name"
            if "Output for configuration" in line:
                parts = line.replace("*", "").replace(":", " ").split()
                # Find the configuration identifier
                try:
                    conf_idx = parts.index("configuration")
                    if conf_idx + 1 < len(parts):
                        # Could be number or name
                        current_name = parts[conf_idx + 1]
                        # If there's a name after the number, use that
                        if conf_idx + 2 < len(parts) and not parts[conf_idx + 2].isdigit():
                            current_name = parts[conf_idx + 2]
                except (ValueError, IndexError):
                    # Fallback: use a generic name
                    current_name = f"config_{i}"

            # Final energy
            if "Total lattice energy" in line and "eV" in line:
                match = re.search(r"=\s*([-\d.]+)", line)
                if match and current_name:
                    results["energies"][current_name] = float(match.group(1))

            # Final gradient norm
            if "Final Gnorm" in line:
                match = re.search(r"=\s*([-\d.]+)", line)
                if match:
                    results["final_gnorm"] = float(match.group(1))

            # Cell parameters (from comparison section)
            if "Comparison of initial and final structures" in line and current_name:
                # Skip to first dashed line
                i += 1
                while i < len(lines) and "-------" not in lines[i]:
                    i += 1
                i += 1  # Skip the first dashed line

                # Skip header line ("Parameter  Initial  Final  Difference  Percent")
                while i < len(lines) and "-------" not in lines[i]:
                    i += 1
                i += 1  # Skip the second dashed line

                # Initialize cell params for this structure
                if current_name not in results["final_cells"]:
                    results["final_cells"][current_name] = {}

                # Parse cell parameters
                # Format: Parameter  Initial  Final  Difference  Percent
                while i < len(lines):
                    cell_line = lines[i].strip()
                    if not cell_line or "-------" in cell_line:
                        break

                    parts = cell_line.split()
                    if len(parts) >= 3:
                        param_name = parts[0].lower()
                        try:
                            # Final value is typically the 3rd column (index 2)
                            # But format varies, so try to get the "Final" column
                            if param_name in ["a", "b", "c"]:
                                final_value = float(parts[2]) if len(parts) > 2 else float(parts[1])
                                results["final_cells"][current_name][param_name] = final_value
                            elif param_name in ["alpha", "beta", "gamma"]:
                                final_value = float(parts[2]) if len(parts) > 2 else float(parts[1])
                                results["final_cells"][current_name][param_name] = final_value
                            elif param_name == "volume":
                                final_value = float(parts[2]) if len(parts) > 2 else float(parts[1])
                                results["final_cells"][current_name]["volume"] = final_value
                        except (ValueError, IndexError):
                            pass
                    i += 1
                continue

            i += 1

        return results

    def _params_to_config(
        self,
        params: np.ndarray,
        base_config: PotentialConfig,
        param_mapping: List[Tuple[str, str, str]],
    ) -> PotentialConfig:
        """Convert flat parameter array to PotentialConfig."""
        config = PotentialConfig(
            name=base_config.name,
            charge_model=base_config.charge_model,
            potential_type=base_config.potential_type,
            fixed_charges=base_config.fixed_charges.copy(),
        )

        # Copy base parameters
        for pair, bp in base_config.buckingham.items():
            config.buckingham[pair] = BuckinghamParams(
                A=bp.A, rho=bp.rho, C=bp.C,
                rmin=bp.rmin, rmax=bp.rmax,
                A_bounds=bp.A_bounds, rho_bounds=bp.rho_bounds, C_bounds=bp.C_bounds,
            )

        for elem, sp in base_config.shell_params.items():
            config.shell_params[elem] = ShellParams(
                element=elem,
                core_charge=sp.core_charge,
                shell_charge=sp.shell_charge,
                spring_k=sp.spring_k,
                spring_k_bounds=sp.spring_k_bounds,
            )

        for elem, qp in base_config.qeq_params.items():
            config.qeq_params[elem] = QEqParams(
                element=elem,
                chi=qp.chi, mu=qp.mu,
                q_min=qp.q_min, q_max=qp.q_max,
                chi_bounds=qp.chi_bounds, mu_bounds=qp.mu_bounds,
            )

        # Update with optimized parameters
        for i, (category, key, attr) in enumerate(param_mapping):
            value = params[i]

            if category == "buckingham":
                pair = tuple(key.split("-"))
                setattr(config.buckingham[pair], attr, value)
            elif category == "shell":
                setattr(config.shell_params[key], attr, value)
            elif category == "qeq":
                setattr(config.qeq_params[key], attr, value)
            elif category == "eem":
                setattr(config.eem_params[key], attr, value)

        return config

    def _config_to_params(
        self,
        config: PotentialConfig,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[str, str, str]]]:
        """
        Extract optimizable parameters from config.

        Returns:
            (params_array, bounds_list, param_mapping)
        """
        params = []
        bounds = []
        mapping = []

        # Buckingham parameters
        for pair, bp in config.buckingham.items():
            pair_key = f"{pair[0]}-{pair[1]}"

            params.append(bp.A)
            bounds.append(bp.A_bounds)
            mapping.append(("buckingham", pair_key, "A"))

            params.append(bp.rho)
            bounds.append(bp.rho_bounds)
            mapping.append(("buckingham", pair_key, "rho"))

            # Optionally fit C
            # params.append(bp.C)
            # bounds.append(bp.C_bounds)
            # mapping.append(("buckingham", pair_key, "C"))

        # Shell parameters (spring constants)
        if config.charge_model == ChargeModel.SHELL and config.fit_charges:
            for elem, sp in config.shell_params.items():
                params.append(sp.spring_k)
                bounds.append(sp.spring_k_bounds)
                mapping.append(("shell", elem, "spring_k"))

        # QEq parameters
        if config.charge_model == ChargeModel.QEQ and config.fit_charges:
            for elem, qp in config.qeq_params.items():
                params.append(qp.chi)
                bounds.append(qp.chi_bounds)
                mapping.append(("qeq", elem, "chi"))

                params.append(qp.mu)
                bounds.append(qp.mu_bounds)
                mapping.append(("qeq", elem, "mu"))

        # EEM parameters
        if config.charge_model == ChargeModel.EEM and config.fit_charges:
            for elem, ep in config.eem_params.items():
                params.append(ep.chi)
                bounds.append(ep.chi_bounds)
                mapping.append(("eem", elem, "chi"))

                params.append(ep.mu)
                bounds.append(ep.mu_bounds)
                mapping.append(("eem", elem, "mu"))

        return np.array(params), bounds, mapping

    def _calculate_objective(
        self,
        params: np.ndarray,
        base_config: PotentialConfig,
        param_mapping: List[Tuple[str, str, str]],
        targets: List[FitTarget],
    ) -> float:
        """Calculate objective function for given parameters."""
        # Check cache
        params_key = tuple(np.round(params, 4))
        if params_key in self._cache:
            return self._cache[params_key]

        self._iteration += 1

        # Build config from params
        config = self._params_to_config(params, base_config, param_mapping)

        # Generate and run GULP
        input_content = self.generate_gulp_input(
            config, targets,
            keywords="conp optimise compare",
            for_fitting=True,
        )

        success, output, results = self.run_gulp(input_content, f"fit_{self._iteration}")

        if not success:
            self._cache[params_key] = self.MAX_OBJECTIVE_VALUE
            return self.MAX_OBJECTIVE_VALUE

        # Calculate objective
        objective = 0.0

        for target in targets:
            name = target.name

            # Energy contribution
            if target.energy is not None and name in results.get("energies", {}):
                calc_energy = results["energies"][name]
                diff = (target.energy - calc_energy) ** 2
                objective += diff * target.energy_weight

            # Lattice parameter contribution
            if target.lattice_params and name in results.get("final_cells", {}):
                for param, ref_value in target.lattice_params.items():
                    if param in results["final_cells"][name]:
                        calc_value = results["final_cells"][name][param]
                        diff = (ref_value - calc_value) ** 2
                        objective += diff * target.lattice_weight

        # Update best
        if objective < self._best_objective:
            self._best_objective = objective
            self._log(f"  Iter {self._iteration}: New best = {objective:.6e}")

        self._history.append(objective)
        self._cache[params_key] = objective

        return objective

    def fit(
        self,
        config: PotentialConfig,
        targets: List[FitTarget],
        method: str = "dual_annealing",
        maxiter: int = 1000,
        **kwargs,
    ) -> FitResult:
        """
        Fit potential parameters to target data.

        Args:
            config: Initial potential configuration
            targets: List of fitting targets
            method: Optimization method ("dual_annealing", "Nelder-Mead", "Powell", "L-BFGS-B")
            maxiter: Maximum iterations
            **kwargs: Additional arguments for optimizer

        Returns:
            FitResult with optimized configuration
        """
        self._log(f"Starting fit with {len(targets)} targets using {method}")
        self._log(f"Charge model: {config.charge_model.value}")

        # Reset state
        self._iteration = 0
        self._best_objective = self.MAX_OBJECTIVE_VALUE
        self._history = []
        self._cache = {}

        # Extract parameters
        initial_params, bounds, param_mapping = self._config_to_params(config)
        self._log(f"Fitting {len(initial_params)} parameters")

        # Define objective function
        def objective(x):
            return self._calculate_objective(x, config, param_mapping, targets)

        # Run optimization
        try:
            if method == "dual_annealing":
                result = dual_annealing(
                    objective,
                    bounds=bounds,
                    x0=initial_params,
                    maxiter=maxiter,
                    seed=42,
                    **kwargs,
                )
            elif method in ["Nelder-Mead", "Powell"]:
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    options={"maxiter": maxiter},
                    **kwargs,
                )
            elif method == "L-BFGS-B":
                result = minimize(
                    objective,
                    initial_params,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": maxiter},
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            converged = result.success if hasattr(result, "success") else True
            final_objective = result.fun
            final_params = result.x

        except Exception as e:
            self._log(f"Optimization failed: {e}")
            return FitResult(
                config=config,
                objective_value=self._best_objective,
                n_iterations=self._iteration,
                converged=False,
                history=self._history,
                message=str(e),
            )

        # Build final config
        final_config = self._params_to_config(final_params, config, param_mapping)

        self._log(f"Fit complete: objective = {final_objective:.6e}, iterations = {self._iteration}")

        return FitResult(
            config=final_config,
            objective_value=final_objective,
            n_iterations=self._iteration,
            converged=converged,
            history=self._history,
            message="Optimization completed successfully",
        )

    def fit_native(
        self,
        config: PotentialConfig,
        targets: List[FitTarget],
        fit_cycles: int = 100,
        fit_method: str = "bfgs",
        relax_structures: bool = True,
        simultaneous: bool = True,
    ) -> FitResult:
        """
        Fit potential using GULP's internal fitting optimizer.

        This is MUCH faster than the external scipy-based fit() method
        because it runs GULP only once with native fitting.

        Args:
            config: Initial potential configuration
            targets: List of fitting targets
            fit_cycles: Maximum fitting cycles (default: 100)
            fit_method: GULP fit method ("bfgs", "dfp", "steep")
            relax_structures: If True, relax structures during fitting
            simultaneous: If True, fit all structures simultaneously

        Returns:
            FitResult with optimized configuration

        Example:
            >>> result = fitter.fit_native(config, targets, fit_cycles=200)
        """
        self._log(f"Starting NATIVE fit with {len(targets)} targets")
        self._log(f"Charge model: {config.charge_model.value}")
        self._log(f"Using GULP internal {fit_method} optimizer")

        # Generate native fit input
        input_content = self._generate_native_fit_input(
            config, targets,
            fit_cycles=fit_cycles,
            fit_method=fit_method,
            relax_structures=relax_structures,
            simultaneous=simultaneous,
        )

        # Run GULP with fitting
        success, output, results = self.run_gulp(input_content, "native_fit")

        if not success:
            self._log("GULP fitting failed")
            return FitResult(
                config=config,
                objective_value=1e10,
                n_iterations=0,
                converged=False,
                history=[],
                message=f"GULP fitting failed: {output[:500] if output else 'No output'}",
            )

        # Parse fitted parameters from output
        fitted_config = self._parse_fitted_parameters(output, config)
        final_objective = self._parse_final_objective(output)
        n_iterations = self._parse_fit_iterations(output)
        converged = "Fit completed" in output or "Conditions for a minimum" in output

        self._log(f"Native fit complete: objective = {final_objective:.6e}")
        self._log(f"Converged: {converged}")

        return FitResult(
            config=fitted_config,
            objective_value=final_objective,
            n_iterations=n_iterations,
            converged=converged,
            history=[],
            message="Native GULP fitting completed",
        )

    def _generate_native_fit_input(
        self,
        config: PotentialConfig,
        targets: List[FitTarget],
        fit_cycles: int = 100,
        fit_method: str = "bfgs",
        relax_structures: bool = True,
        simultaneous: bool = True,
    ) -> str:
        """Generate GULP input file for native fitting."""
        # Build keywords
        keywords = ["fit", "conp"]
        if simultaneous and len(targets) > 1:
            keywords.append("simul")
        if relax_structures:
            keywords.append("opti")
        
        # Add charge model keyword
        if config.charge_model == ChargeModel.QEQ:
            keywords.append("qeq")
        elif config.charge_model == ChargeModel.EEM:
            keywords.append("eem")

        all_elements = set()
        for target in targets:
            all_elements.update(target.atoms.get_chemical_symbols())
        all_elements = sorted(all_elements)

        sections = [" ".join(keywords)]

        # Fit cycles
        sections.append(f"cycles fit {fit_cycles}")

        # Add each structure with observables
        for target in targets:
            sections.append(self.generate_structure_block(target, config))
            sections.append(self._generate_fit_observables(target))
            sections.append("")

        # Species block
        sections.append(self.generate_species_block(config, all_elements))

        # Potential blocks with fitting flags
        sections.append(self._generate_buckingham_with_flags(config))

        # Charge model specific blocks with fitting flags
        if config.charge_model == ChargeModel.SHELL:
            sections.append(self._generate_spring_with_flags(config))
        elif config.charge_model == ChargeModel.QEQ:
            sections.append(self._generate_qeq_with_flags(config))

        # Output settings
        sections.append("dump native_fit.grs")

        return "\n\n".join(sections)

    def _generate_fit_observables(self, target: FitTarget) -> str:
        """Generate observables block for GULP fitting."""
        lines = ["observables"]

        if target.energy is not None:
            lines.append("energy eV")
            lines.append(f"  {target.energy:.6f} {target.energy_weight:.4f}")

        if target.lattice_params:
            for param, value in target.lattice_params.items():
                lines.append(f"{param} {value:.6f} {target.lattice_weight:.4f}")

        lines.append("end")
        return "\n".join(lines)

    def _generate_buckingham_with_flags(self, config: PotentialConfig) -> str:
        """Generate Buckingham block with GULP fitting flags (0=fixed, 1=fit)."""
        lines = ["buckingham"]

        for (elem1, elem2), bp in config.buckingham.items():
            type1 = "shel" if config.charge_model == ChargeModel.SHELL and elem1 in config.shell_params else "core"
            type2 = "shel" if config.charge_model == ChargeModel.SHELL and elem2 in config.shell_params else "core"

            # Fitting flags: 1 1 0 means fit A and rho, not C
            fit_A = 1 if config.fit_buckingham else 0
            fit_rho = 1 if config.fit_buckingham else 0
            fit_C = 0  # Usually keep C fixed

            lines.append(
                f"{elem1} {type1} {elem2} {type2} "
                f"{bp.A:.6f} {bp.rho:.6f} {bp.C:.6f} "
                f"{bp.rmin:.2f} {bp.rmax:.2f} "
                f"{fit_A} {fit_rho} {fit_C}"
            )

        return "\n".join(lines)

    def _generate_spring_with_flags(self, config: PotentialConfig) -> str:
        """Generate spring block with fitting flags."""
        if config.charge_model != ChargeModel.SHELL:
            return ""

        lines = ["spring"]
        for elem, sp in config.shell_params.items():
            fit_flag = 1 if config.fit_charges else 0
            lines.append(f"{elem} {sp.spring_k:.6f} {fit_flag}")

        return "\n".join(lines)

    def _generate_qeq_with_flags(self, config: PotentialConfig) -> str:
        """Generate QEq block with fitting flags."""
        if config.charge_model != ChargeModel.QEQ:
            return ""

        lines = ["qeq"]
        for elem, qp in config.qeq_params.items():
            fit_chi = 1 if config.fit_charges else 0
            fit_mu = 1 if config.fit_charges else 0
            # GULP QEq format: element chi mu qmin qmax fit_chi fit_mu
            lines.append(
                f"{elem} {qp.chi:.6f} {qp.mu:.6f} "
                f"{qp.q_min:.2f} {qp.q_max:.2f} "
                f"{fit_chi} {fit_mu}"
            )

        return "\n".join(lines)

    def _parse_fitted_parameters(self, output: str, base_config: PotentialConfig) -> PotentialConfig:
        """Parse fitted parameters from GULP output."""
        import re
        
        # Create new config
        config = PotentialConfig(
            name=base_config.name + "_fitted",
            charge_model=base_config.charge_model,
            potential_type=base_config.potential_type,
            fixed_charges=base_config.fixed_charges.copy(),
            fit_buckingham=base_config.fit_buckingham,
            fit_charges=base_config.fit_charges,
        )

        # Parse Buckingham parameters from output
        # Look for "Final values of general potential parameters :"
        buck_section = False
        lines = output.split("\n")
        
        for i, line in enumerate(lines):
            if "Final values of" in line and "parameter" in line.lower():
                buck_section = True
                continue
            
            if buck_section and "----" in line and i > 0:
                buck_section = False
                continue

            if buck_section:
                # Try to parse parameter lines
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Look for Buckingham A and rho values
                        # Format varies, try to extract numbers
                        pass
                    except:
                        pass

        # Fallback: copy base config parameters (GULP writes to .grs file)
        # Try to read from .grs restart file
        grs_file = os.path.join(self.working_dir, "native_fit.grs")
        if os.path.exists(grs_file):
            config = self._parse_grs_file(grs_file, base_config)
        else:
            # Copy base config
            config.buckingham = {k: BuckinghamParams(
                A=v.A, rho=v.rho, C=v.C,
                rmin=v.rmin, rmax=v.rmax,
                A_bounds=v.A_bounds, rho_bounds=v.rho_bounds, C_bounds=v.C_bounds
            ) for k, v in base_config.buckingham.items()}
            config.shell_params = base_config.shell_params.copy()
            config.qeq_params = base_config.qeq_params.copy()

        return config

    def _parse_grs_file(self, grs_file: str, base_config: PotentialConfig) -> PotentialConfig:
        """Parse GULP restart file (.grs) for fitted parameters."""
        import re

        config = PotentialConfig(
            name=base_config.name + "_fitted",
            charge_model=base_config.charge_model,
            potential_type=base_config.potential_type,
            fixed_charges=base_config.fixed_charges.copy(),
        )

        with open(grs_file, "r") as f:
            content = f.read()
            lines = content.split("\n")

        in_buck = False
        in_spring = False
        in_qeq = False

        for line in lines:
            line = line.strip()
            
            if line.startswith("buckingham"):
                in_buck = True
                in_spring = False
                in_qeq = False
                continue
            elif line.startswith("spring"):
                in_buck = False
                in_spring = True
                in_qeq = False
                continue
            elif line.startswith("qeq"):
                in_buck = False
                in_spring = False
                in_qeq = True
                continue
            elif line.startswith(("species", "cell", "frac", "observ", "end", "#")):
                in_buck = False
                in_spring = False
                in_qeq = False
                continue

            if in_buck and line:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        elem1, type1, elem2, type2 = parts[0], parts[1], parts[2], parts[3]
                        A, rho, C = float(parts[4]), float(parts[5]), float(parts[6])
                        pair = (elem1, elem2)
                        if pair in base_config.buckingham or (elem2, elem1) in base_config.buckingham:
                            if pair not in base_config.buckingham:
                                pair = (elem2, elem1)
                            config.buckingham[pair] = BuckinghamParams(
                                A=A, rho=rho, C=C,
                                rmin=base_config.buckingham[pair].rmin,
                                rmax=base_config.buckingham[pair].rmax,
                            )
                    except (ValueError, IndexError):
                        pass

            if in_spring and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        elem = parts[0]
                        k = float(parts[1])
                        if elem in base_config.shell_params:
                            sp = base_config.shell_params[elem]
                            config.shell_params[elem] = ShellParams(
                                element=elem,
                                core_charge=sp.core_charge,
                                shell_charge=sp.shell_charge,
                                spring_k=k,
                            )
                    except (ValueError, IndexError):
                        pass

            if in_qeq and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        elem = parts[0]
                        chi, mu = float(parts[1]), float(parts[2])
                        if elem in base_config.qeq_params:
                            qp = base_config.qeq_params[elem]
                            config.qeq_params[elem] = QEqParams(
                                element=elem,
                                chi=chi, mu=mu,
                                q_min=qp.q_min, q_max=qp.q_max,
                            )
                    except (ValueError, IndexError):
                        pass

        # Fallback for any missing params
        for pair, bp in base_config.buckingham.items():
            if pair not in config.buckingham:
                config.buckingham[pair] = bp
        for elem, sp in base_config.shell_params.items():
            if elem not in config.shell_params:
                config.shell_params[elem] = sp
        for elem, qp in base_config.qeq_params.items():
            if elem not in config.qeq_params:
                config.qeq_params[elem] = qp

        return config

    def _parse_final_objective(self, output: str) -> float:
        """Parse final sum of squares from GULP output."""
        import re
        
        # Look for "Sum of squares" value
        patterns = [
            r"Sum of squares\s*=\s*([\d.eE+-]+)",
            r"Final sum of squares\s*=\s*([\d.eE+-]+)",
            r"Fitting\s+Sum of squares\s*=\s*([\d.eE+-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))
        
        return 1e10

    def _parse_fit_iterations(self, output: str) -> int:
        """Parse number of fitting iterations from output."""
        import re
        match = re.search(r"Cycle\s*:\s*(\d+)", output)
        if match:
            return int(match.group(1))
        return 0


    def save_potential(
        self,
        config: PotentialConfig,
        output_file: str,
        output_format: str = "gulp_lib",
    ) -> None:
        """
        Save fitted potential to file.

        Args:
            config: Potential configuration
            output_file: Output file path
            output_format: Output format ("gulp_lib", "json")
        """
        if output_format == "json":
            with open(output_file, "w") as f:
                json.dump(config.to_dict(), f, indent=2)

        elif output_format == "gulp_lib":
            # Generate GULP library format
            lines = [f"# {config.name}", f"# Charge model: {config.charge_model.value}", ""]

            # Species
            all_elements = set()
            for pair in config.buckingham.keys():
                all_elements.update(pair)

            lines.append("# Species charges")
            if config.charge_model == ChargeModel.SHELL:
                for elem in sorted(all_elements):
                    if elem in config.shell_params:
                        sp = config.shell_params[elem]
                        lines.append(f"# {elem}: core={sp.core_charge:.4f}, shell={sp.shell_charge:.4f}")
                    elif elem in config.fixed_charges:
                        lines.append(f"# {elem}: {config.fixed_charges[elem]:.4f}")

            lines.append("")
            lines.append("buckingham")
            for (elem1, elem2), bp in config.buckingham.items():
                type1 = "shel" if elem1 in config.shell_params else "core"
                type2 = "shel" if elem2 in config.shell_params else "core"
                lines.append(
                    f"{elem1} {type1} {elem2} {type2} "
                    f"{bp.A:.6f} {bp.rho:.6f} {bp.C:.6f} {bp.rmin:.1f} {bp.rmax:.1f}"
                )

            if config.charge_model == ChargeModel.SHELL:
                lines.append("")
                lines.append("spring")
                for elem, sp in config.shell_params.items():
                    lines.append(f"{elem} {sp.spring_k:.6f}")

            if config.charge_model == ChargeModel.QEQ:
                lines.append("")
                lines.append("qeq")
                for elem, qp in config.qeq_params.items():
                    lines.append(f"{elem} {qp.chi:.6f} {qp.mu:.6f} {qp.q_min:.2f} {qp.q_max:.2f}")

            with open(output_file, "w") as f:
                f.write("\n".join(lines))

        self._log(f"Saved potential to {output_file}")

    def load_potential(self, input_file: str) -> PotentialConfig:
        """Load potential configuration from JSON file."""
        with open(input_file, "r") as f:
            data = json.load(f)
        return PotentialConfig.from_dict(data)

    def cleanup(self) -> None:
        """Clean up working directory."""
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)


# =============================================================================
# Convenience functions
# =============================================================================

def create_nbwo_shell_config(
    initial_params: Optional[Dict] = None,
) -> PotentialConfig:
    """
    Create initial Buckingham + Shell configuration for Nb-W-O system.

    Uses literature values as starting point.
    """
    config = PotentialConfig(
        name="NbWO_shell",
        charge_model=ChargeModel.SHELL,
        potential_type=PotentialType.BUCKINGHAM,
    )

    # Default Buckingham parameters (based on similar oxides)
    config.buckingham = {
        ("Nb", "O"): BuckinghamParams(A=1500.0, rho=0.35, C=0.0),
        ("W", "O"): BuckinghamParams(A=1800.0, rho=0.34, C=0.0),
        ("O", "O"): BuckinghamParams(A=22764.0, rho=0.149, C=27.88),
    }

    # Shell parameters for oxygen (typical values)
    config.shell_params = {
        "O": ShellParams(
            element="O",
            core_charge=0.9,
            shell_charge=-2.9,
            spring_k=74.92,
        ),
    }

    # Fixed charges for metals
    config.fixed_charges = {
        "Nb": 5.0,
        "W": 6.0,
    }

    # Override with user params if provided
    if initial_params:
        for key, value in initial_params.items():
            if key in config.buckingham:
                for attr, val in value.items():
                    setattr(config.buckingham[key], attr, val)

    return config


def create_nbwo_qeq_config(
    initial_params: Optional[Dict] = None,
) -> PotentialConfig:
    """
    Create initial Buckingham + QEq configuration for Nb-W-O system.

    QEq allows charges to vary based on local environment.
    """
    config = PotentialConfig(
        name="NbWO_qeq",
        charge_model=ChargeModel.QEQ,
        potential_type=PotentialType.BUCKINGHAM,
    )

    # Buckingham parameters (may need adjustment for QEq)
    config.buckingham = {
        ("Nb", "O"): BuckinghamParams(A=1200.0, rho=0.36, C=0.0),
        ("W", "O"): BuckinghamParams(A=1500.0, rho=0.35, C=0.0),
        ("O", "O"): BuckinghamParams(A=20000.0, rho=0.16, C=25.0),
    }

    # QEq parameters (electronegativity, hardness)
    # Values from literature / GULP eqeq.lib
    config.qeq_params = {
        "Nb": QEqParams(element="Nb", chi=3.8865, mu=2.9935, q_min=0.0, q_max=5.0),
        "W": QEqParams(element="W", chi=4.3975, mu=3.5825, q_min=0.0, q_max=6.0),
        "O": QEqParams(element="O", chi=8.7410, mu=6.0824, q_min=-2.5, q_max=0.0),
    }

    # Initial charges for optimization starting point
    config.fixed_charges = {
        "Nb": 2.5,
        "W": 3.0,
        "O": -1.0,
    }

    config.fit_charges = True  # Enable QEq parameter fitting

    return config


def fit_from_training_data(
    training_data: List,  # List of TrainingData from data_collector
    charge_model: str = "shell",
    method: str = "dual_annealing",
    maxiter: int = 500,
    gulp_command: Optional[str] = None,
    output_file: Optional[str] = None,
) -> FitResult:
    """
    Fit GULP potential from training data collected by DataCollector.

    Args:
        training_data: List of TrainingData objects
        charge_model: "shell" or "qeq"
        method: Optimization method
        maxiter: Maximum iterations
        gulp_command: Path to GULP executable
        output_file: Save fitted potential to file

    Returns:
        FitResult with optimized potential
    """
    # Create fitter
    fitter = GULPFitter(gulp_command=gulp_command)

    # Create initial config
    if charge_model == "shell":
        config = create_nbwo_shell_config()
    elif charge_model == "qeq":
        config = create_nbwo_qeq_config()
    else:
        raise ValueError(f"Unknown charge model: {charge_model}")

    # Convert training data to FitTargets
    targets = []
    for i, td in enumerate(training_data):
        if td.energy is None:
            continue

        # Get lattice parameters
        cell = td.structure.get_cell()
        a, b, c = cell.lengths()
        alpha, beta, gamma = cell.angles()

        target = FitTarget(
            name=f"struct_{i}",
            atoms=td.structure,
            energy=td.energy,
            forces=td.forces,
            lattice_params={"a": a, "b": b, "c": c},
        )
        targets.append(target)

    print(f"Fitting potential to {len(targets)} structures")

    # Run fitting
    result = fitter.fit(config, targets, method=method, maxiter=maxiter)

    # Save if requested
    if output_file and result.converged:
        fitter.save_potential(result.config, output_file, output_format="gulp_lib")
        fitter.save_potential(result.config, output_file.replace(".lib", ".json"), output_format="json")

    fitter.cleanup()

    return result
