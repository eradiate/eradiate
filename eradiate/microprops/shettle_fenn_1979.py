"""
Aerosols models according to :cite:`Shettle1979ModelsAerosolsLower`.
"""
from typing import Callable

import numpy as np

from ..units import unit_registry as ureg


@ureg.wraps(
    ret=None,
    args=("micrometer", "cm^-3 * micrometer^-1", "dimensionless"),
    strict=False,
)
def size_distribution(mr: np.ndarray, nd: np.ndarray, std: np.ndarray) -> Callable:
    """
    Return a function that evaluate the particle cumulative number density per
    unit of radius.

    This function implements the equation (1) in
    :cite:`Shettle1979ModelsAerosolsLower`.

    Parameter ``mr`` (:class:`~numpy.ndarray`):
        Mode radius [micrometer].

    Parameter ``nd`` (:class:`~numpy.ndarray`):
        Number density per unit of radius [cm^-3 * micrometer^-1].

    Parameter ``std`` (:class:`~numpy.ndarray`):
        Standard deviation [dimensionless].

    Returns → Callable:
        A function :class:`~numpy.ndarray` → :class:`~numpy.ndarray` that
        evaluate the particle cumulative number density per unit of radius.
    """

    return lambda r: (nd / np.log(10) * r * std * np.sqrt(2)) * np.exp(
        -(np.square(np.log10(r) - np.log10(mr))) / (2 * np.square(std))
    )
