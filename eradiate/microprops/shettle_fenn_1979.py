"""
Aerosols models according to :cite:`Shettle1979ModelsAerosolsLower`.
"""
import enum
from typing import Callable, Union

import numpy as np
import pint
import xarray as xr

from ..units import unit_registry as ureg


class AerosolModel(enum.Enum):
    """
    Aerosol model enumeration.
    """

    RURAL = "rural"
    URBAN = "urban"
    MARITIME = "maritime"
    TROPOSPHERIC = "tropospheric"


class AerosolComponent(enum.Enum):
    """
    Aerosol component enumeration.
    """

    WATER_SOLUBLE = "water_soluble"
    DUST_LIKE = "dust_like"
    SOOT_LIKE = "soot_like"
    SEA_SALT = "sea_salt"
    WATER = "water"


# aerosol models' compositions (for refractive index computation)
COMPOSITION = {
    AerosolModel.RURAL: {
        AerosolComponent.WATER_SOLUBLE: 0.7,
        AerosolComponent.DUST_LIKE: 0.3,
    },
    AerosolModel.URBAN: {
        AerosolComponent.WATER_SOLUBLE: 0.7 * 0.8,
        AerosolComponent.DUST_LIKE: 0.3 * 0.8,
        AerosolComponent.SOOT_LIKE: 1 * 0.2,
    },
    AerosolModel.MARITIME: {
        AerosolComponent.WATER_SOLUBLE: "variable",  # particles of continental origin
        AerosolComponent.DUST_LIKE: "variable",  # particles of continental origin
        AerosolComponent.SEA_SALT: "variable",  # particles of oceanic origin
    },
}

SIZE_DISTRIBUTION_PARAMS = {
    AerosolModel.RURAL: (
        [0.999875, 0.000125],  # n
        [0.35 * ureg.dimensionless, 0.4 * ureg.dimensionless],  # std
    )
}


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
    r1: float, r2: float, n1: float, n2: float, s1: float, s2: float
) -> Callable:
    """
    Compute the aerosol size distribution according to equation (1) of
    :cite:`Shettle1979ModelsAerosolsLower`.
    """
    return lambda r: n1 * lognorm(r0=r1, std=s1)(r) + n2 * lognorm(r0=r2, std=s2)(r)


@ureg.wraps(ret=None, args=("nm", "micrometer", "micrometer", None, None), strict=False)
def wet_aeorosl_refractive_index(
    w: Union[pint.Quantity, np.ndarray, float],
    r0: Union[pint.Quantity, float],
    rw: Union[pint.Quantity, float],
    n0: xr.DataArray,
    nw: xr.DataArray,
) -> xr.DataArray:
    """
    Compute the wet aerosol particle refractive index according to equation (6)
    of :cite:`Shettle1979ModelsAerosolsLower`.

    Parameter ``w`` (:class:`~pint.Quantity` or :class:`~numpy.ndarray` or float)
        Wavelength [nm].

    Parameter ``r0`` (:class:`~pint.Quantity` or float):
        Dry particle size [micrometer].

    Parameter ``rw`` (:class:`~pint.Quantity` or float):
        Wet particle size [micrometer].

    Parameter ``n0`` (:class:`~xarray.DataArray`):
        Dry particle refractive index data.

    Parameter ``nw`` (:class:`~xarray.DataArray`):
        Water refractive index data.

    Returns → class:`~xarray.DataArray`:
        Wet aerosol particle refractive index.
    """
    return nw.interp(w=w) + (n0.interp(w=w) - nw.interp(w=w)) * np.power(r0 / rw, 3)
