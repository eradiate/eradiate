"""Absorption cross section spectrum data sets shipped with Eradiate.
"""
import enum

import numpy as np
import xarray as xr

from .core import DataGetter
from .. import path_resolver as _presolver
from .._units import unit_registry as ureg


class Absorber(enum.Enum):
    """
    Absorbing species enumeration.
    """

    us76_u86_4 = "us76_u86_4"
    CH4 = "CH4"
    CO = "CO"
    CO2 = "CO2"
    H2O = "H2O"
    N2O = "N2O"
    O2 = "O2"
    O3 = "O3"


class Engine(enum.Enum):
    """
    Absorption cross section computation engine enumeration.
    """

    SPECTRA = "spectra"


class Group(enum.Enum):
    """
    Data sets group enumeration.

    .. note::
        A group refers to a collection of data sets that differ by their
        wavenumber bins.
    """

    SPECTRA_US76_U86_4 = (Absorber.us76_u86_4, Engine.SPECTRA)
    SPECTRA_CH4 = (Absorber.CH4, Engine.SPECTRA)
    SPECTRA_CO = (Absorber.CO, Engine.SPECTRA)
    SPECTRA_CO2 = (Absorber.CO2, Engine.SPECTRA)
    SPECTRA_H2O = (Absorber.H2O, Engine.SPECTRA)
    SPECTRA_N2O = (Absorber.N2O, Engine.SPECTRA)
    SPECTRA_O2 = (Absorber.O2, Engine.SPECTRA)
    SPECTRA_O3 = (Absorber.O3, Engine.SPECTRA)


def _resolve_group(absorber, engine):
    """
    Return the group corresponding to the (absorber, engine) pair, if it
    exists.
    """
    for group in Group:
        if group.value == (absorber, engine):
            return group
    raise ValueError(f"Cannot resolve group ({absorber}, {engine})")


# fmt: off
_WAVENUMBER_BINS = {
    Group.SPECTRA_US76_U86_4: [(x, x + 1000)
                              for x in np.arange(4000, 25000, 1000)] +
                              [(25000, 25711)],
    Group.SPECTRA_CH4: [(x, x + 100) for x in np.arange(4000, 11500, 100)] +
                       [(11500, 11502)],
    Group.SPECTRA_CO: [(x, x + 100) for x in np.arange(4000, 14400, 100)] +
                      [(14400, 14478)],
    Group.SPECTRA_CO2: [(x, x + 100) for x in np.arange(4000, 14000, 100)] +
                       [(14000 , 14076)],
    Group.SPECTRA_H2O: [(x, x + 100) for x in np.arange(4000, 25700, 100)] +
                       [(25700, 25711)],
    Group.SPECTRA_N2O: [(x, x + 100) for x in np.arange(4000, 10300, 100)] +
                       [(10300, 10364)],
    Group.SPECTRA_O2: [(x, x + 100) for x in np.arange(4000, 17200, 100)] +
                      [(17200, 17273)],
    Group.SPECTRA_O3: [(x, x + 100) for x in np.arange(4000, 6900, 100)] +
                      [(6900, 6997)],
}
# fmt: on


def _resolve_w_bin(group, wavenumber):
    """
    Return the wavenumber bin corresponding to a group and a wavenumber value,
    if it exists.
    """
    for w_bin in _WAVENUMBER_BINS[group]:
        w_min, w_max = w_bin
        if w_min <= wavenumber <= w_max:
            return w_bin
    raise ValueError(
        f"Cannot find wavenumber bin corresponding to wavenumber "
        f"value {wavenumber} in for the group {group}"
    )


def _get_data_set_id(group, w_bin):
    """
    Return the data set identifier of the data set specified by a group and
    a wavenumber bin.
    """
    absorber, engine = group.value
    w_min, w_max = w_bin
    return f"{absorber.value}-{engine.value}-{w_min}_{w_max}"


_ROOT_DIR = "spectra/absorption"

_PATHS = {}
for group in Group:
    absorber, engine = group.value
    for wavenumber_bin in _WAVENUMBER_BINS[group]:
        data_set_id = _get_data_set_id(group, wavenumber_bin)
        _PATHS[data_set_id] = f"{_ROOT_DIR}/{absorber.value}/{data_set_id}.nc"


class _AbsorptionGetter(DataGetter):
    PATHS = _PATHS

    @classmethod
    def open(cls, id):
        path = cls.path(id)
        paths = _presolver.glob(path)

        try:
            return xr.open_mfdataset(paths)
        except OSError as e:
            raise OSError(f"while opening file at {path}: {str(e)}")

    @classmethod
    def find(cls):
        result = {}

        for id in cls.PATHS.keys():
            paths = _presolver.glob(cls.path(id))
            result[id] = bool(len(list(paths)))

        return result


@ureg.wraps(ret=None, args=("cm^-1", None, None), strict=False)
def find_dataset(wavenumber, absorber, engine):
    """
    Find the dataset corresponding to a given wavenumber,
    absorber and absorption cross section engine.

    Parameter ``wavenumber`` (:class:`~pint.Quantity`):
        Wavenumber value [cm^-1].

    Parameter ``absorber`` (:class:`Absorber`):
        Absorber.

    Parameter ``engine`` (:class:`Engine`):
        Engine used to compute the absorption cross sections.

    Returns → :class:`str`:
        Available dataset id.
    """
    group = _resolve_group(absorber=absorber, engine=engine)
    w_bin = _resolve_w_bin(group=group, wavenumber=wavenumber)
    return _get_data_set_id(group=group, w_bin=w_bin)
