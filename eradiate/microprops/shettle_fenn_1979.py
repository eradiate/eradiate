"""
Aerosols models according to :cite:`Shettle1979ModelsAerosolsLower`.
"""
import numpy as np
from scipy.stats import lognorm

from ..units import unit_registry as ureg


def lognorm(r: np.ndarray, ri: float, si: float, ni: float = 1.0):
    """
    Compute the log-normal distribution as in equation (1) of
    :cite:`Shettle1979ModelsAerosolsLower`.
    """
    return (ni / (np.log(10) * r * si * np.sqrt(2 * np.pi))) * np.exp(
        -np.square(np.log10(r) - np.log10(ri)) / (2 * np.square(si))
    )


@ureg.wraps(
    ret=None,
    args=(
        "micrometer",
        "micrometer",
        "micrometer",
        "cm^-3",
        "cm^-3",
        "dimensionless",
        "dimensionless",
    ),
    strict=False,
)
def size_distribution(
    r: np.ndarray, r1: float, r2: float, n1: float, n2: float, s1: float, s2: float
) -> np.ndarray:
    """
    Compute the aerosol size distribution according to equation (1) of
    :cite:`Shettle1979ModelsAerosolsLower`.
    """
    return lognorm(r=r, ri=r1, si=s1, ni=n1) + lognorm(r=r, ri=r2, si=s2, ni=n2)
