"""Absorption cross section spectrum data sets shipped with Eradiate.

**Category ID**: ``absorption_cross_section_spectrum``

.. list-table:: Available data sets and corresponding identifiers
   :widths: 1 1 1 1
   :header-rows: 1

   * - Data set ID
     - Reference
     - Spectral range [cm^-1]
     - Spectral range [nm]
   * - ``usa_mls-fullrange``
     - ``eradiate-datasets_maker/scripts/spectra/acs/spectra/usa_mls.py``
     - [18000, 19000]
     - [~527, ~555]
   * - ``usa_mls-fullrange``
     - ``eradiate-datasets_maker/scripts/spectra/acs/spectra/usa_mls.py``
     - [4000, 25711]
     - [~389, 2500]
   * - ``us76_u86_4-fullrange``
     - ``eradiate-datasets_maker/scripts/spectra/acs/spectra/us76_u86_4.py``
     - [4000, 25711]
     - [~389, 2500]
"""

import xarray as xr

from .core import DataGetter
from ..util.presolver import PathResolver

_presolver = PathResolver()


class _AbsorptionGetter(DataGetter):
    _PATHS = {
        "usa_mls-narrowrange": "spectra/absorption/usa_mls/narrowrange/*.nc",
        "usa_mls-fullrange": "spectra/absorption/usa_mls/fullrange/*.nc",
        "us76_u86_4-fullrange": "spectra/absorption/us76_u86_4/fullrange/*.nc",
        "test": "tests/absorption/us76_u86_4/*.nc"
    }

    @classmethod
    def open(cls, id):
        paths = _presolver.glob(cls.path(id))
        return xr.open_mfdataset(paths)

    @classmethod
    def find(cls):
        result = {}

        for id in cls._PATHS.keys():
            paths = _presolver.glob(cls.path(id))
            result[id] = bool(len(list(paths)))

        return result
