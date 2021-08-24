"""
Aerosols models according to :cite:`Shettle1979ModelsAerosolsLower`.
"""
from typing import Callable

import numpy as np

from ..units import unit_registry as ureg


@ureg.wraps(ret=None, args=("micrometer", "dimensionless"), strict=False)
def lognorm(r0: float, std: float = 0.4) -> Callable:
    """
    Return a log-normal distribution as in equation (1) of
    :cite:`Shettle1979ModelsAerosolsLower`.

    Parameter ``r0`` (float):
        Mode radius [micrometer].

    Parameter ``s`` (float):
        Standard deviation of the distribution.

    Returns → Callable:
        A function (:class:`numpy.ndarray` → :class:`numpy.ndarray`)
        that evaluates the log-normal distribution.
    """
    return lambda r: (1.0 / (np.log(10) * r * std * np.sqrt(2 * np.pi))) * np.exp(
        -np.square(np.log10(r) - np.log10(r0)) / (2 * np.square(std))
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
    return lognorm(r0=r1, std=s1, n=n1)(r) + lognorm(r0=r2, std=s2, n=n2)(r)
