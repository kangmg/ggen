"""
GULP Calculator with parallel execution support for ggen.

Wraps ASE's built-in GULP calculator with:
- Process isolation via temporary directories
- Multiprocessing for concurrent job execution
- Configurable CPU resource allocation
"""

import os
import tempfile
import shutil
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
from ase import Atoms
from ase.calculators.gulp import GULP
from ase.calculators.calculator import Calculator, all_changes


def _get_cpu_count() -> int:
    """Get available CPU count."""
    return os.cpu_count() or 1


def _run_gulp_isolated(
    atoms: Atoms,
    gulp_command: str,
    gulp_lib: str,
    keywords: str,
    library: Optional[str],
    options: List[str],
    shel: List[str],
) -> Dict[str, Any]:
    """
    Run single GULP calculation in isolated temporary directory.

    This function is designed to be called from multiprocessing pool.
    All parameters must be picklable (no Calculator objects).
    """
    results = {"energy": None, "forces": None, "stress": None, "error": None}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set environment for this process
            env = os.environ.copy()
            env["GULP_LIB"] = gulp_lib
            os.environ["GULP_LIB"] = gulp_lib

            # Create calculator in isolated directory
            calc = GULP(
                label=os.path.join(tmpdir, "gulp"),
                keywords=keywords,
                library=library,
                options=options,
                shel=shel,
            )

            # Set command with proper paths
            # Format: "gulp < PREFIX.gin > PREFIX.got"
            calc.command = gulp_command

            # Run calculation
            atoms = atoms.copy()
            atoms.calc = calc

            results["energy"] = atoms.get_potential_energy()

            if "forces" in calc.results:
                results["forces"] = calc.results["forces"]

            if "stress" in calc.results:
                results["stress"] = calc.results["stress"]

    except Exception as e:
        results["error"] = str(e)

    return results


class GULPCalculator(Calculator):
    """
    ASE Calculator wrapper for GULP with parallel execution support.

    Wraps ASE's built-in GULP calculator and adds:
    - Process isolation (each calculation in separate temp directory)
    - Multiprocessing for batch calculations
    - Configurable resource allocation

    Args:
        gulp_command: GULP command template
            (default: "gulp < PREFIX.gin > PREFIX.got")
        gulp_lib: Path to GULP library directory
        keywords: GULP keywords (default: "conp gradients")
        library: Force field library file (e.g., "catlow.lib")
        options: Additional GULP options as list of strings
        shel: List of elements with shell model
        n_workers: Number of concurrent GULP processes (default: auto)
        total_cores: Total cores available (default: all)

    Example:
        # Single calculation
        calc = GULPCalculator(
            gulp_command="/path/to/gulp < PREFIX.gin > PREFIX.got",
            gulp_lib="/path/to/Libraries",
            library="catlow.lib",
        )
        atoms.calc = calc
        energy = atoms.get_potential_energy()

        # Batch parallel calculation
        calc = GULPCalculator(..., n_workers=8)
        results = calc.calculate_batch(atoms_list)
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        gulp_command: Optional[str] = None,
        gulp_lib: Optional[str] = None,
        keywords: str = "conp gradients",
        library: Optional[str] = None,
        options: Optional[List[str]] = None,
        shel: Optional[List[str]] = None,
        n_workers: Optional[int] = None,
        total_cores: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # GULP configuration
        self.gulp_lib = gulp_lib or os.environ.get("GULP_LIB", "")

        if gulp_command:
            self.gulp_command = gulp_command
        else:
            gulp_exe = os.environ.get("ASE_GULP_COMMAND")
            if gulp_exe:
                self.gulp_command = gulp_exe
            else:
                self.gulp_command = "gulp < PREFIX.gin > PREFIX.got"

        self.keywords = keywords
        self.library = library
        self.options = options or []
        self.shel = shel or []

        # Parallelization settings
        self.total_cores = total_cores or _get_cpu_count()
        self.n_workers = n_workers if n_workers is not None else self.total_cores

        # Validate
        if self.n_workers > self.total_cores:
            raise ValueError(
                f"n_workers ({self.n_workers}) exceeds total_cores ({self.total_cores})"
            )

        # Thread pool (created lazily)
        # Using ThreadPoolExecutor because GULP runs as subprocess
        # Python GIL doesn't affect subprocess execution - they run in parallel
        self._executor: Optional[ThreadPoolExecutor] = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        return self._executor

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = None,
    ) -> None:
        """
        Run GULP calculation for single structure.

        Uses isolated temporary directory to avoid file conflicts.
        """
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes

        Calculator.calculate(self, atoms, properties, system_changes)

        # Run in isolated environment
        result = _run_gulp_isolated(
            atoms=self.atoms,
            gulp_command=self.gulp_command,
            gulp_lib=self.gulp_lib,
            keywords=self.keywords,
            library=self.library,
            options=self.options,
            shel=self.shel,
        )

        if result["error"]:
            raise RuntimeError(f"GULP calculation failed: {result['error']}")

        # Store results
        if result["energy"] is not None:
            self.results["energy"] = result["energy"]
        else:
            raise RuntimeError("GULP did not return energy")

        if result["forces"] is not None:
            self.results["forces"] = result["forces"]

        if result["stress"] is not None:
            self.results["stress"] = result["stress"]

    def calculate_batch(
        self,
        atoms_list: List[Atoms],
        properties: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run GULP calculations for multiple structures in parallel.

        Uses process pool with n_workers concurrent GULP processes,
        each in isolated temporary directories.

        Args:
            atoms_list: List of ASE Atoms objects
            properties: Properties to calculate (default: all)

        Returns:
            List of result dicts with 'energy', 'forces', 'stress', 'error' keys
        """
        if properties is None:
            properties = self.implemented_properties

        executor = self._get_executor()

        # Submit all jobs
        futures = []
        for atoms in atoms_list:
            future = executor.submit(
                _run_gulp_isolated,
                atoms=atoms,
                gulp_command=self.gulp_command,
                gulp_lib=self.gulp_lib,
                keywords=self.keywords,
                library=self.library,
                options=self.options,
                shel=self.shel,
            )
            futures.append(future)

        # Collect results in order
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "energy": None,
                    "forces": None,
                    "stress": None,
                })

        return results

    def __del__(self):
        """Cleanup executor on deletion."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        return (
            f"GULPCalculator(n_workers={self.n_workers}, "
            f"total_cores={self.total_cores}, "
            f"library={self.library})"
        )


def get_gulp_calculator(
    gulp_exe: Optional[str] = None,
    gulp_lib: Optional[str] = None,
    library: Optional[str] = None,
    keywords: str = "conp gradients",
    options: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    **kwargs,
) -> GULPCalculator:
    """
    Factory function to create GULP calculator.

    Looks for environment variables if paths not provided:
    - ASE_GULP_COMMAND or GULP_EXE: Path to GULP executable
    - GULP_LIB: Path to GULP library directory

    Args:
        gulp_exe: Path to GULP executable
        gulp_lib: Path to GULP library directory
        library: Force field library file name
        keywords: GULP keywords
        options: Additional GULP options
        n_workers: Concurrent GULP processes

    Returns:
        Configured GULPCalculator instance
    """
    # Build command
    if gulp_exe:
        gulp_command = f"{gulp_exe} < PREFIX.gin > PREFIX.got"
    else:
        gulp_command = os.environ.get("ASE_GULP_COMMAND")
        if not gulp_command:
            gulp_exe = os.environ.get("GULP_EXE")
            if gulp_exe:
                gulp_command = f"{gulp_exe} < PREFIX.gin > PREFIX.got"
            else:
                gulp_command = "gulp < PREFIX.gin > PREFIX.got"

    if gulp_lib is None:
        gulp_lib = os.environ.get("GULP_LIB", "")

    return GULPCalculator(
        gulp_command=gulp_command,
        gulp_lib=gulp_lib,
        keywords=keywords,
        library=library,
        options=options,
        n_workers=n_workers,
        **kwargs,
    )
