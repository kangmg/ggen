"""
GGen: Crystal Generation and Mutation Library

A powerful Python library for crystal structure generation, mutation, and evolutionary optimization.
Built on top of PyXtal, pymatgen, and ASE, GGen provides an intuitive interface for generating,
modifying, and analyzing crystal structures with built-in energy evaluation using ORB models.
"""

from .calculator import get_orb_calculator
from .colors import Colors
from .database import ExplorationRun, StoredStructure, StructureDatabase
from .explorer import CandidateResult, ChemistryExplorer, ExplorationResult
from .ggen import GGen, get_space_group_cache_info, clear_space_group_cache
from .operations import MutationError, Operations
from .phonons import (
    PhononResult,
    StabilityTestResult,
    calculate_phonons,
    check_dynamical_stability,
    find_first_stable_candidate,
    select_stable_candidate,
)
from .report import SystemExplorer, SystemReport, StabilityStats, SpaceGroupStats
from .utils import parse_chemical_formula

# GULP integration
from .gulp_calculator import GULPCalculator, get_gulp_calculator, get_gfnff_calculator
from .data_collector import DataCollector, TrainingData, collect_from_cif_files
from .robust_pipeline import (
    RobustPipeline,
    PipelineConfig,
    SamplingConfig,
    FilterConfig,
    DataValidator,
    DiversityFilter,
    CheckpointManager,
    run_pipeline,
)

# GULP Potential Fitting
from .gulp_fitter import (
    GULPFitter,
    PotentialConfig,
    ChargeModel,
    PotentialType,
    BuckinghamParams,
    ShellParams,
    QEqParams,
    EEMParams,
    FitTarget,
    FitResult,
    create_nbwo_shell_config,
    create_nbwo_qeq_config,
    fit_from_training_data,
)

# MLIP Data Collection with ASE DB
from .mlip_collector import (
    MLIPCollector,
    CollectionConfig,
    CollectionMode,
    CollectionStats,
    collect_from_cif_directory,
    collect_for_gulp_fitting,
)

__version__ = "0.1.0"
__author__ = "Matt Moderwell"
__email__ = "matt@ouro.foundation"

__all__ = [
    # Core
    "GGen",
    # Explorer
    "ChemistryExplorer",
    "CandidateResult",
    "ExplorationResult",
    # Database
    "StructureDatabase",
    "StoredStructure",
    "ExplorationRun",
    # Reporting / Analysis
    "SystemExplorer",
    "SystemReport",
    "StabilityStats",
    "SpaceGroupStats",
    # Operations
    "Operations",
    "MutationError",
    # Phonons / Dynamical Stability
    "PhononResult",
    "StabilityTestResult",
    "calculate_phonons",
    "check_dynamical_stability",
    "find_first_stable_candidate",
    "select_stable_candidate",
    # Utilities
    "Colors",
    "get_orb_calculator",
    "parse_chemical_formula",
    # Cache management
    "get_space_group_cache_info",
    "clear_space_group_cache",
    # GULP Calculator
    "GULPCalculator",
    "get_gulp_calculator",
    "get_gfnff_calculator",
    # Data Collection
    "DataCollector",
    "TrainingData",
    "collect_from_cif_files",
    # Robust Pipeline
    "RobustPipeline",
    "PipelineConfig",
    "SamplingConfig",
    "FilterConfig",
    "DataValidator",
    "DiversityFilter",
    "CheckpointManager",
    "run_pipeline",
    # GULP Potential Fitting
    "GULPFitter",
    "PotentialConfig",
    "ChargeModel",
    "PotentialType",
    "BuckinghamParams",
    "ShellParams",
    "QEqParams",
    "EEMParams",
    "FitTarget",
    "FitResult",
    "create_nbwo_shell_config",
    "create_nbwo_qeq_config",
    "fit_from_training_data",
    # MLIP Data Collection
    "MLIPCollector",
    "CollectionConfig",
    "CollectionMode",
    "CollectionStats",
    "collect_from_cif_directory",
    "collect_for_gulp_fitting",
]
