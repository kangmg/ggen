# QEq Parameters Database
#
# Source: Open Babel qeq.txt
# Charge equilibration parameters for all elements
#
# Parameters:
#   chi: Electronegativity (eV)
#   mu: Chemical hardness (eV)
#   radius: Screening radius (Å)

from typing import Dict, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class QEqElementParams:
    """QEq parameters for a single element."""
    chi: float      # Electronegativity (eV)
    mu: float       # Chemical hardness (eV)
    radius: float   # Screening radius (Å)


# Full QEq parameter database (Open Babel)
QEQ_PARAMS_DATABASE: Dict[str, QEqElementParams] = {
    "H":  QEqElementParams(chi=4.528,  mu=13.8904, radius=0.8271),
    "He": QEqElementParams(chi=9.66,   mu=29.8400, radius=1.7640),
    "Li": QEqElementParams(chi=3.006,  mu=4.7720,  radius=2.4076),
    "Be": QEqElementParams(chi=4.877,  mu=8.8860,  radius=1.2930),
    "B":  QEqElementParams(chi=5.11,   mu=9.5000,  radius=1.2094),
    "C":  QEqElementParams(chi=5.343,  mu=10.1260, radius=1.1346),
    "N":  QEqElementParams(chi=6.899,  mu=11.7600, radius=0.9770),
    "O":  QEqElementParams(chi=8.741,  mu=13.3640, radius=0.8597),
    "F":  QEqElementParams(chi=10.874, mu=14.9480, radius=0.7686),
    "Ne": QEqElementParams(chi=11.04,  mu=21.1000, radius=1.4394),
    "Na": QEqElementParams(chi=2.843,  mu=4.5920,  radius=2.5020),
    "Mg": QEqElementParams(chi=3.951,  mu=7.3860,  radius=1.5555),
    "Al": QEqElementParams(chi=4.06,   mu=7.1800,  radius=1.6002),
    "Si": QEqElementParams(chi=4.168,  mu=6.9740,  radius=1.6474),
    "P":  QEqElementParams(chi=5.463,  mu=8.0000,  radius=1.4362),
    "S":  QEqElementParams(chi=6.928,  mu=8.9720,  radius=1.2806),
    "Cl": QEqElementParams(chi=8.564,  mu=9.8920,  radius=1.1615),
    "Ar": QEqElementParams(chi=9.465,  mu=12.7100, radius=1.2259),
    "K":  QEqElementParams(chi=2.421,  mu=3.8400,  radius=2.9920),
    "Ca": QEqElementParams(chi=3.231,  mu=5.7600,  radius=1.9947),
    "Sc": QEqElementParams(chi=3.395,  mu=6.1600,  radius=1.8651),
    "Ti": QEqElementParams(chi=3.47,   mu=6.7600,  radius=1.6996),
    "V":  QEqElementParams(chi=3.65,   mu=6.8200,  radius=1.6846),
    "Cr": QEqElementParams(chi=3.415,  mu=7.7300,  radius=1.4863),
    "Mn": QEqElementParams(chi=3.325,  mu=8.2100,  radius=1.3994),
    "Fe": QEqElementParams(chi=3.76,   mu=8.2800,  radius=1.3876),
    "Co": QEqElementParams(chi=4.105,  mu=8.3500,  radius=1.3760),
    "Ni": QEqElementParams(chi=4.465,  mu=8.4100,  radius=1.3661),
    "Cu": QEqElementParams(chi=4.2,    mu=8.4400,  radius=1.3613),
    "Zn": QEqElementParams(chi=5.106,  mu=8.5700,  radius=1.3406),
    "Ga": QEqElementParams(chi=3.641,  mu=6.3200,  radius=1.8179),
    "Ge": QEqElementParams(chi=4.051,  mu=6.8760,  radius=1.6709),
    "As": QEqElementParams(chi=5.188,  mu=7.6180,  radius=1.5082),
    "Se": QEqElementParams(chi=6.428,  mu=8.2620,  radius=1.3906),
    "Br": QEqElementParams(chi=7.79,   mu=8.8500,  radius=1.2982),
    "Kr": QEqElementParams(chi=8.505,  mu=11.4300, radius=1.0268),
    "Rb": QEqElementParams(chi=2.331,  mu=3.6920,  radius=3.1119),
    "Sr": QEqElementParams(chi=3.024,  mu=4.8800,  radius=2.3544),
    "Y":  QEqElementParams(chi=3.83,   mu=5.6200,  radius=2.0444),
    "Zr": QEqElementParams(chi=3.4,    mu=7.1000,  radius=1.6182),
    "Nb": QEqElementParams(chi=3.55,   mu=6.7600,  radius=1.6996),
    "Mo": QEqElementParams(chi=3.465,  mu=7.5100,  radius=1.5299),
    "Tc": QEqElementParams(chi=3.29,   mu=7.9800,  radius=1.4398),
    "Ru": QEqElementParams(chi=3.575,  mu=8.0300,  radius=1.4308),
    "Rh": QEqElementParams(chi=3.975,  mu=8.0100,  radius=1.4344),
    "Pd": QEqElementParams(chi=4.32,   mu=8.0000,  radius=1.4362),
    "Ag": QEqElementParams(chi=4.436,  mu=6.2680,  radius=1.8330),
    "Cd": QEqElementParams(chi=5.034,  mu=7.9140,  radius=1.4518),
    "In": QEqElementParams(chi=3.506,  mu=5.7920,  radius=1.9836),
    "Sn": QEqElementParams(chi=3.987,  mu=6.2480,  radius=1.8389),
    "Sb": QEqElementParams(chi=4.899,  mu=6.6840,  radius=1.7189),
    "Te": QEqElementParams(chi=5.816,  mu=7.0520,  radius=1.6292),
    "I":  QEqElementParams(chi=6.822,  mu=7.5240,  radius=1.5270),
    "Xe": QEqElementParams(chi=7.595,  mu=9.9500,  radius=1.1547),
    "Cs": QEqElementParams(chi=2.183,  mu=3.4220,  radius=3.3575),
    "Ba": QEqElementParams(chi=2.814,  mu=4.7920,  radius=2.3976),
    "La": QEqElementParams(chi=2.8355, mu=5.4830,  radius=2.0954),
    "Ce": QEqElementParams(chi=2.774,  mu=5.3840,  radius=2.1340),
    "Pr": QEqElementParams(chi=2.858,  mu=5.1280,  radius=2.2405),
    "Nd": QEqElementParams(chi=2.8685, mu=5.2410,  radius=2.1922),
    "Pm": QEqElementParams(chi=2.881,  mu=5.3460,  radius=2.1491),
    "Sm": QEqElementParams(chi=2.9115, mu=5.4390,  radius=2.1124),
    "Eu": QEqElementParams(chi=2.8785, mu=5.5750,  radius=2.0609),
    "Gd": QEqElementParams(chi=3.1665, mu=5.9490,  radius=1.9313),
    "Tb": QEqElementParams(chi=3.018,  mu=5.6680,  radius=2.0270),
    "Dy": QEqElementParams(chi=3.0555, mu=5.7430,  radius=2.0006),
    "Ho": QEqElementParams(chi=3.127,  mu=5.7820,  radius=1.9871),
    "Er": QEqElementParams(chi=3.1865, mu=5.8290,  radius=1.9711),
    "Tm": QEqElementParams(chi=3.2514, mu=5.8658,  radius=1.9587),
    "Yb": QEqElementParams(chi=3.2889, mu=5.9300,  radius=1.9375),
    "Lu": QEqElementParams(chi=2.9629, mu=4.9258,  radius=2.3325),
    "Hf": QEqElementParams(chi=3.7,    mu=6.8000,  radius=1.6896),
    "Ta": QEqElementParams(chi=5.1,    mu=5.7000,  radius=2.0157),
    "W":  QEqElementParams(chi=4.63,   mu=6.6200,  radius=1.7355),
    "Re": QEqElementParams(chi=3.96,   mu=7.8400,  radius=1.4655),
    "Os": QEqElementParams(chi=5.14,   mu=7.2600,  radius=1.5825),
    "Ir": QEqElementParams(chi=5.00,   mu=8.0000,  radius=1.4362),
    "Pt": QEqElementParams(chi=4.79,   mu=8.8600,  radius=1.2968),
    "Au": QEqElementParams(chi=4.894,  mu=5.1720,  radius=2.2214),
    "Hg": QEqElementParams(chi=6.27,   mu=8.3200,  radius=1.3809),
    "Tl": QEqElementParams(chi=3.2,    mu=5.8000,  radius=1.9809),
    "Pb": QEqElementParams(chi=3.9,    mu=7.0600,  radius=1.6274),
    "Bi": QEqElementParams(chi=4.69,   mu=7.4800,  radius=1.5360),
    "Po": QEqElementParams(chi=4.21,   mu=8.4200,  radius=1.3645),
    "At": QEqElementParams(chi=4.75,   mu=9.5000,  radius=1.2094),
    "Rn": QEqElementParams(chi=5.37,   mu=10.7400, radius=1.0698),
    "Fr": QEqElementParams(chi=2.00,   mu=4.0000,  radius=2.8723),
    "Ra": QEqElementParams(chi=2.843,  mu=4.8680,  radius=2.3602),
    "Ac": QEqElementParams(chi=2.835,  mu=5.6700,  radius=2.0263),
    "Th": QEqElementParams(chi=3.175,  mu=5.8100,  radius=1.9775),
    "Pa": QEqElementParams(chi=2.985,  mu=5.8100,  radius=1.9775),
    "U":  QEqElementParams(chi=3.341,  mu=5.7060,  radius=2.0135),
    "Np": QEqElementParams(chi=3.549,  mu=5.4340,  radius=2.1143),
    "Pu": QEqElementParams(chi=3.243,  mu=5.6380,  radius=2.0378),
    "Am": QEqElementParams(chi=2.9895, mu=6.0070,  radius=1.9126),
    "Cm": QEqElementParams(chi=2.8315, mu=6.3790,  radius=1.8011),
    "Bk": QEqElementParams(chi=3.1935, mu=6.0710,  radius=1.8925),
    "Cf": QEqElementParams(chi=3.197,  mu=6.2020,  radius=1.8525),
    "Es": QEqElementParams(chi=3.333,  mu=6.1780,  radius=1.8597),
    "Fm": QEqElementParams(chi=3.4,    mu=6.2000,  radius=1.8531),
    "Md": QEqElementParams(chi=3.47,   mu=6.2200,  radius=1.8471),
    "No": QEqElementParams(chi=3.475,  mu=6.3500,  radius=1.8093),
    "Lr": QEqElementParams(chi=3.5,    mu=6.4000,  radius=1.7952),
}


def get_qeq_params(element: str) -> Optional[QEqElementParams]:
    """
    Get QEq parameters for an element.
    
    Args:
        element: Element symbol (e.g., "Nb", "O")
    
    Returns:
        QEqElementParams or None if not found
    """
    return QEQ_PARAMS_DATABASE.get(element)


def get_qeq_params_for_elements(elements: list) -> Dict[str, QEqElementParams]:
    """
    Get QEq parameters for multiple elements.
    
    Args:
        elements: List of element symbols
    
    Returns:
        Dict mapping element to QEqElementParams
    
    Raises:
        KeyError: If any element is not in database
    """
    result = {}
    for el in elements:
        if el not in QEQ_PARAMS_DATABASE:
            raise KeyError(f"No QEq parameters for element: {el}")
        result[el] = QEQ_PARAMS_DATABASE[el]
    return result


def build_qeq_config(elements: list) -> dict:
    """
    Build QEqParams dict for GULPFitter from element list.
    
    Args:
        elements: List of element symbols (e.g., ["Nb", "W", "O"])
    
    Returns:
        Dict suitable for PotentialConfig.qeq_params
    
    Example:
        >>> qeq_params = build_qeq_config(["Nb", "W", "O"])
        >>> config = PotentialConfig(qeq_params=qeq_params, ...)
    """
    from .gulp_fitter import QEqParams
    
    result = {}
    for el in elements:
        if el not in QEQ_PARAMS_DATABASE:
            raise KeyError(f"No QEq parameters for element: {el}")
        db_params = QEQ_PARAMS_DATABASE[el]
        result[el] = QEqParams(
            element=el,
            chi=db_params.chi,
            mu=db_params.mu,
        )
    return result
