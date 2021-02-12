"""Absorption cross section spectrum data sets shipped with Eradiate.

**Category ID**: ``absorption_spectrum``

.. list-table:: Available data sets and corresponding identifiers
   :widths: 1 1 1
   :header-rows: 1

   * - Dataset ID
     - Reference
     - Spetral range [nm]
   * - ``spectra-us76_u86_4``
     - ``eradiate-datasets_maker/scripts/spectra/acs/spectra/us76_u86_4.py``
     - [~389, 2500]
"""

import numpy as np
import xarray as xr

from .core import DataGetter
from ..util.presolver import PathResolver
from ..util.units import ureg

_presolver = PathResolver()

_US76_U86_4_PATH = "spectra/absorption/us76_u86_4"
_US76_U86_4_PREF = "spectra-us76_u86_4"


class _AbsorptionGetter(DataGetter):
    PATHS = {
        **{f"{_US76_U86_4_PREF}-{x}_{x + 500}": f"{_US76_U86_4_PATH}/"
                                                f"{_US76_U86_4_PREF}-{x}_"
                                                f"{x + 500}/*.nc"
           for x in np.arange(4000, 25500, 500)},
        f"{_US76_U86_4_PREF}-25500_25711": f"{_US76_U86_4_PATH}/"
                                           f"{_US76_U86_4_PREF}-"
                                           f"25500_25711/*.nc",
        "test": "tests/absorption/us76_u86_4/*.nc",
    }

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
def find_dataset(wavenumber, absorber="us76_u86_4", engine="spectra"):
    """Finds the dataset corresponding to a given wavenumber,
    absorber and absorption cross section engine.

    Parameter ``wavenumber`` (:class:`~pint.Quantity`):
        Wavenumber value [cm^-1].

    Parameter ``absorber`` (str):
        Absorber name.

    Parameter ``engine`` (str):
        Engine used to compute the absorption cross sections.

    Returns → str:
        Available dataset id.
    """
    if absorber == "us76_u86_4":
        if engine != "spectra":
            raise ValueError(f"engine {engine} is not supported.")
        for d in [_AbsorptionGetter.PATHS[k] for k in _AbsorptionGetter.PATHS
                  if k != "test"]:
            path = _presolver.resolve(d.strip("/*.nc"))
            if path.is_absolute():
                _engine, _absorber, w_range = tuple(path.name.split("-"))
                if _absorber == absorber and _engine == engine:
                    w_min, w_max = w_range.split("_")
                    if float(w_min) <= wavenumber < float(w_max):
                        return path.name
        raise ValueError(f"could not find the dataset corresponding to "
                         f"wavenumber = {wavenumber}, "
                         f"absorber = {absorber} and "
                         f"engine = {engine}")
    else:
        raise ValueError(f"absorber {absorber} is not supported.")
