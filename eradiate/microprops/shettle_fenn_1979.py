"""
Aerosols models according to :cite:`Shettle1979ModelsAerosolsLower`.
"""
import enum
from typing import Callable, Union

import numpy as np
import pint
import xarray as xr

from .. import data
from ..units import to_quantity
from ..units import unit_registry as ureg


class AerosolModel(enum.Enum):
    """
    Aerosol model enumeration.
    """

    RURAL = "rural"
    URBAN = "urban"
    MARITIME = "maritime"
    TROPOSPHERIC = "tropospheric"


@ureg.wraps(ret=None, args=("micrometer", ""), strict=False)
def lognorm(r0: float, std: float = 0.4) -> Callable:
    """
    Return a log-normal distribution as in equation (1) of
    :cite:`Shettle1979ModelsAerosolsLower`.

    Parameter ``r0`` (float):
        Mode radius [micrometer].

    Parameter ``s`` (float):
        Standard deviation of the distribution [].

    Returns → Callable:
        A function (:class:`numpy.ndarray` → :class:`numpy.ndarray`)
        that evaluates the log-normal distribution.
    """
    return lambda r: (1.0 / (np.log(10) * r * std * np.sqrt(2 * np.pi))) * np.exp(
        -np.square(np.log10(r) - np.log10(r0)) / (2 * np.square(std))
    )


# ------------------------------------------------------------------------------
#                           Rural aerosol model
# ------------------------------------------------------------------------------


@ureg.wraps(ret=None, args=(""), strict=False)
def rural_aerosol_large_particles_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the large particles in the rural aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_2")
    r2 = to_quantity(ds.r.sel(model="rural", mode=2).interp(rh=rh))
    n2 = ureg.Quantity(0.000125, "cm^-3")
    s2 = ureg.Quantity(0.4 * ureg.dimensionless)
    return n2 * lognorm(r0=r2, std=s2)


@ureg.wraps(ret=None, args=(""), strict=False)
def rural_aerosol_small_particles_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the small particles in the rural aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_2")
    r1 = to_quantity(ds.r.sel(model="rural", mode=1).interp(rh=rh))
    n1 = ureg.Quantity(0.999875, "cm^-3")
    s1 = ureg.Quantity(0.35 * ureg.dimensionless)
    return n1 * lognorm(r0=r1, std=s1)


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def rural_aerosol_small_particles_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the small particles in the rural aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_4a")
    refractive_index = ds.eta_r + 1j * ds.eta_i
    return to_quantity(refractive_index.interp(w=w.to(ds.w.units), rh=rh))


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def rural_aerosol_large_particles_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the large particles in the rural aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_4b")
    refractive_index = ds.eta_r + 1j * ds.eta_i
    return to_quantity(refractive_index.interp(w=w.to(ds.w.units), rh=rh))


# ------------------------------------------------------------------------------
#                           Urban aerosol model
# ------------------------------------------------------------------------------


@ureg.wraps(ret=None, args=(""), strict=False)
def urban_aerosol_large_particles_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the large particles in the urban aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_2")
    r2 = to_quantity(ds.r.sel(model="urban", mode=2).interp(rh=rh))
    n2 = ureg.Quantity(0.000125, "cm^-3")
    s2 = ureg.Quantity(0.4 * ureg.dimensionless)
    return n2 * lognorm(r0=r2, std=s2)


@ureg.wraps(ret=None, args=(""), strict=False)
def urban_aerosol_small_particles_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the small particles in the urban aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_2")
    r1 = to_quantity(ds.r.sel(model="urban", mode=1).interp(rh=rh))
    n1 = ureg.Quantity(0.999875, "cm^-3")
    s1 = ureg.Quantity(0.35 * ureg.dimensionless)
    return n1 * lognorm(r0=r1, std=s1)


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def urban_aerosol_small_particles_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the small particles in the urban aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_5a")
    refractive_index = ds.eta_r + 1j * ds.eta_i
    return to_quantity(refractive_index.interp(w=w.to(ds.w.units), rh=rh))


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def urban_aerosol_large_particles_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the large particles in the urban aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_5b")
    refractive_index = ds.eta_r + 1j * ds.eta_i
    return to_quantity(refractive_index.interp(w=w.to(ds.w.units), rh=rh))


# ------------------------------------------------------------------------------
#                           Maritime aerosol model
# ------------------------------------------------------------------------------


@ureg.wraps(ret=None, args=(""), strict=False)
def maritime_aerosol_oceanic_component_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the oceanic component in the maritime
    aerosol model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_2")
    r2 = to_quantity(ds.r.sel(model="maritime", mode=2).interp(rh=rh))
    s2 = ureg.Quantity(0.4 * ureg.dimensionless)
    return lognorm(r0=r2, std=s2)


@ureg.wraps(ret=None, args=(""), strict=False)
def maritime_aerosol_continental_component_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the continental component in the maritime
    aerosol model.
    """
    return rural_aerosol_small_particles_radius_distribution(rh=rh)


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def maritime_aerosol_oceanic_component_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the oceanic component in the maritime aerosol
    model.
    """
    ds = data.open(category="microprops", id="shettle_fenn_1979_table_6")
    refractive_index = ds.eta_r + 1j * ds.eta_i
    return to_quantity(refractive_index.interp(w=w.to(ds.w.units), rh=rh))


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def maritime_aerosol_continental_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the continental component in the maritime
    aerosol model.
    """
    return rural_aerosol_small_particles_refractive_index(w=w, rh=rh)


# ------------------------------------------------------------------------------
#                           Tropospheric aerosol model
# ------------------------------------------------------------------------------


@ureg.wraps(ret=None, args=(""), strict=False)
def tropospheric_aerosol_radius_distribution(
    rh: Union[pint.Quantity, float] = 0.75
) -> Callable:
    """
    Return the radius distribution of the particles in the tropospheric aerosol
    model.
    """
    return rural_aerosol_small_particles_radius_distribution(rh=rh)


@ureg.wraps(ret=None, args=("nm", ""), strict=False)
def tropospheric_aerosol_refractive_index(
    w: Union[pint.Quantity, float], rh: Union[pint.Quantity, float] = 0.75
) -> pint.Quantity:
    """
    Return the refractive index of the particles in the tropospheric aerosol
    model.
    """
    return rural_aerosol_small_particles_refractive_index(w=w, rh=rh)
