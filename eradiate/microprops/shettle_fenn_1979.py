"""
Aerosols models according to :cite:`Shettle1979ModelsAerosolsLower`.
"""
from typing import Callable

import numpy as np

from ..units import unit_registry as ureg


def lognorm(ri: float, si: float, ni: float = 1.0) -> Callable:
    """
    Return a log-normal distribution as in equation (1) of
    :cite:`Shettle1979ModelsAerosolsLower`.
    """
    return lambda r: (ni / (np.log(10) * r * si * np.sqrt(2 * np.pi))) * np.exp(
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
    return lognorm(ri=r1, si=s1, ni=n1)(r) + lognorm(ri=r2, si=s2, ni=n2)(r)
