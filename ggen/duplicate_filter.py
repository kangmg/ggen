"""
Duplicate Structure Filter for Dataset Curation.

Filters out similar structures from training datasets using
structural fingerprints/descriptors to ensure diversity.

Supported methods:
- SOAP (Smooth Overlap of Atomic Positions) via dscribe
- SineCoulombMatrix via matminer
- Energy-based simple filtering
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
import numpy as np
import hashlib
import logging

from ase import Atoms
from ase.db import connect

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for duplicate filtering."""
    method: Literal["soap", "coulomb", "energy"] = "soap"
    threshold: float = 0.95  # Cosine similarity threshold
    
    # SOAP parameters
    soap_r_cut: float = 6.0
    soap_n_max: int = 8
    soap_l_max: int = 6
    soap_sigma: float = 0.3
    soap_average: str = "inner"  # "inner", "outer", or None
    
    # Coulomb matrix parameters
    coulomb_n_atoms: Optional[int] = None  # Max atoms (auto if None)
    
    # Energy filtering
    energy_tolerance: float = 0.001  # eV/atom
    
    # General
    batch_size: int = 100
    verbose: bool = True


@dataclass
class FilterResult:
    """Result of filtering operation."""
    total_structures: int
    unique_structures: int
    removed_structures: int
    unique_ids: List[int]
    removed_ids: List[int]
    method: str
    threshold: float
    
    @property
    def removal_rate(self) -> float:
        return self.removed_structures / self.total_structures if self.total_structures > 0 else 0.0
    
    def summary(self) -> str:
        return (
            f"Filter Results ({self.method}):\n"
            f"  Total: {self.total_structures}\n"
            f"  Unique: {self.unique_structures}\n"
            f"  Removed: {self.removed_structures} ({self.removal_rate:.1%})\n"
            f"  Threshold: {self.threshold}"
        )


class DuplicateFilter:
    """
    Filter duplicate/similar structures from training datasets.
    
    Uses structural fingerprints to identify and remove structures
    that are too similar, improving dataset diversity.
    
    Example:
        >>> filter = DuplicateFilter(method="soap", threshold=0.95)
        >>> result = filter.filter_database("training.db")
        >>> print(result.summary())
        
        # Or filter a list of atoms
        >>> unique_atoms = filter.filter_atoms(atoms_list)
    """
    
    def __init__(
        self,
        method: Literal["soap", "coulomb", "energy"] = "soap",
        threshold: float = 0.95,
        config: Optional[FilterConfig] = None,
    ):
        """
        Initialize duplicate filter.
        
        Args:
            method: Fingerprint method ("soap", "coulomb", "energy")
            threshold: Similarity threshold (0-1). Higher = more strict filtering
            config: Optional detailed configuration
        """
        self.config = config or FilterConfig(method=method, threshold=threshold)
        self.method = self.config.method
        self.threshold = self.config.threshold
        
        self._soap = None
        self._coulomb = None
        self._species = None
    
    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[DuplicateFilter] {msg}")
    
    def _init_soap(self, species: List[str]) -> None:
        """Initialize SOAP descriptor."""
        try:
            from dscribe.descriptors import SOAP
        except ImportError:
            raise ImportError(
                "dscribe is required for SOAP fingerprints. "
                "Install with: pip install dscribe"
            )
        
        self._species = sorted(set(species))
        self._soap = SOAP(
            species=self._species,
            r_cut=self.config.soap_r_cut,
            n_max=self.config.soap_n_max,
            l_max=self.config.soap_l_max,
            sigma=self.config.soap_sigma,
            periodic=True,
            sparse=False,
            average=self.config.soap_average,
        )
        self._log(f"SOAP initialized: species={self._species}, r_cut={self.config.soap_r_cut}")
    
    def _init_coulomb(self, n_atoms: int) -> None:
        """Initialize Sine Coulomb Matrix."""
        try:
            from matminer.featurizers.structure import SineCoulombMatrix
        except ImportError:
            raise ImportError(
                "matminer is required for Coulomb matrix fingerprints. "
                "Install with: pip install matminer"
            )
        
        max_atoms = self.config.coulomb_n_atoms or n_atoms
        self._coulomb = SineCoulombMatrix(n_atoms=max_atoms, flatten=True)
        self._log(f"SineCoulombMatrix initialized: n_atoms={max_atoms}")
    
    def get_fingerprint(self, atoms: Atoms) -> np.ndarray:
        """
        Compute fingerprint for a structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            1D numpy array fingerprint
        """
        if self.method == "soap":
            return self._get_soap_fingerprint(atoms)
        elif self.method == "coulomb":
            return self._get_coulomb_fingerprint(atoms)
        elif self.method == "energy":
            return self._get_energy_fingerprint(atoms)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_soap_fingerprint(self, atoms: Atoms) -> np.ndarray:
        """Compute SOAP fingerprint."""
        if self._soap is None:
            species = list(set(atoms.get_chemical_symbols()))
            self._init_soap(species)
        
        # Check if all species are known
        atom_species = set(atoms.get_chemical_symbols())
        if not atom_species.issubset(set(self._species)):
            # Reinitialize with new species
            self._init_soap(list(atom_species | set(self._species)))
        
        descriptor = self._soap.create(atoms)
        return descriptor.flatten()
    
    def _get_coulomb_fingerprint(self, atoms: Atoms) -> np.ndarray:
        """Compute Sine Coulomb Matrix fingerprint."""
        if self._coulomb is None:
            self._init_coulomb(len(atoms))
        
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            structure = AseAtomsAdaptor.get_structure(atoms)
            features = self._coulomb.featurize(structure)
            return np.array(features)
        except Exception as e:
            self._log(f"Coulomb matrix failed: {e}")
            return np.zeros(1)
    
    def _get_energy_fingerprint(self, atoms: Atoms) -> np.ndarray:
        """Simple energy-based fingerprint."""
        # Use composition + cell volume as proxy
        formula = atoms.get_chemical_formula()
        volume = atoms.get_volume()
        n_atoms = len(atoms)
        
        # Create a simple hash-based vector
        hash_val = int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)
        return np.array([hash_val, volume, n_atoms, volume / n_atoms])
    
    def compute_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Compute cosine similarity between two fingerprints.
        
        Args:
            fp1, fp2: Fingerprint vectors
            
        Returns:
            Similarity score (0-1)
        """
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(fp1, fp2) / (norm1 * norm2)
    
    def filter_atoms(
        self,
        atoms_list: List[Atoms],
        return_indices: bool = False,
    ) -> Union[List[Atoms], tuple]:
        """
        Filter a list of atoms, removing similar structures.
        
        Args:
            atoms_list: List of ASE Atoms
            return_indices: If True, also return indices of kept structures
            
        Returns:
            List of unique atoms, optionally with indices
        """
        if not atoms_list:
            return ([], []) if return_indices else []
        
        self._log(f"Filtering {len(atoms_list)} structures (method={self.method})")
        
        # Compute all fingerprints
        fingerprints = []
        for atoms in atoms_list:
            try:
                fp = self.get_fingerprint(atoms)
                fingerprints.append(fp)
            except Exception as e:
                self._log(f"Fingerprint failed: {e}")
                fingerprints.append(None)
        
        # Greedy filtering
        unique_indices = []
        unique_fps = []
        
        for i, fp in enumerate(fingerprints):
            if fp is None:
                continue
            
            is_unique = True
            for ref_fp in unique_fps:
                sim = self.compute_similarity(fp, ref_fp)
                if sim > self.threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_indices.append(i)
                unique_fps.append(fp)
        
        unique_atoms = [atoms_list[i] for i in unique_indices]
        
        self._log(f"Kept {len(unique_atoms)}/{len(atoms_list)} structures")
        
        if return_indices:
            return unique_atoms, unique_indices
        return unique_atoms
    
    def filter_database(
        self,
        db_path: Union[str, Path],
        output_db_path: Optional[Union[str, Path]] = None,
        copy_unique: bool = True,
    ) -> FilterResult:
        """
        Filter an ASE database, removing similar structures.
        
        Args:
            db_path: Path to input ASE database
            output_db_path: Path for filtered database (optional)
            copy_unique: If True, copy unique structures to output_db
            
        Returns:
            FilterResult with statistics
        """
        db = connect(db_path)
        total = len(db)
        
        self._log(f"Filtering database: {db_path} ({total} structures)")
        
        # Load all structures
        atoms_list = []
        ids = []
        for row in db.select():
            atoms_list.append(row.toatoms())
            ids.append(row.id)
        
        # Filter
        _, unique_indices = self.filter_atoms(atoms_list, return_indices=True)
        
        unique_ids = [ids[i] for i in unique_indices]
        removed_ids = [id_ for id_ in ids if id_ not in unique_ids]
        
        # Optionally copy to new database
        if output_db_path and copy_unique:
            output_db = connect(output_db_path)
            for idx in unique_indices:
                row = db.get(ids[idx])
                atoms = row.toatoms()
                output_db.write(atoms, **row.key_value_pairs)
            self._log(f"Written {len(unique_ids)} structures to {output_db_path}")
        
        result = FilterResult(
            total_structures=total,
            unique_structures=len(unique_ids),
            removed_structures=len(removed_ids),
            unique_ids=unique_ids,
            removed_ids=removed_ids,
            method=self.method,
            threshold=self.threshold,
        )
        
        self._log(result.summary())
        return result
    
    def get_similarity_matrix(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Args:
            atoms_list: List of structures
            
        Returns:
            N x N similarity matrix
        """
        n = len(atoms_list)
        fingerprints = [self.get_fingerprint(a) for a in atoms_list]
        
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self.compute_similarity(fingerprints[i], fingerprints[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix


def filter_training_data(
    db_path: str,
    output_path: Optional[str] = None,
    method: str = "soap",
    threshold: float = 0.95,
) -> FilterResult:
    """
    Convenience function to filter a training database.
    
    Args:
        db_path: Input database path
        output_path: Output database path (optional)
        method: Filtering method ("soap", "coulomb", "energy")
        threshold: Similarity threshold
        
    Returns:
        FilterResult
        
    Example:
        >>> result = filter_training_data("training.db", "filtered.db", threshold=0.90)
        >>> print(f"Kept {result.unique_structures} unique structures")
    """
    filter = DuplicateFilter(method=method, threshold=threshold)
    return filter.filter_database(db_path, output_path)
