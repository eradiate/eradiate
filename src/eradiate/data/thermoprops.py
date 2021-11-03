"""
Atmospheric thermophysical properties profiles data sets shipped with Eradiate.
"""
import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver


class _ThermoPropsProfilesGetter(DataGetter):
    PATHS = {
        "afgl_1986-midlatitude_summer": "thermoprops/afgl_1986-midlatitude_summer.nc",
        "afgl_1986-midlatitude_winter": "thermoprops/afgl_1986-midlatitude_winter.nc",
        "afgl_1986-subarctic_summer": "thermoprops/afgl_1986-subarctic_summer.nc",
        "afgl_1986-subarctic_winter": "thermoprops/afgl_1986-subarctic_winter.nc",
        "afgl_1986-tropical": "thermoprops/afgl_1986-tropical.nc",
        "afgl_1986-us_standard": "thermoprops/afgl_1986-us_standard.nc",
    }

    @classmethod
    def open(cls, id):
        path = _presolver.resolve(cls.path(id))
        return xr.open_dataset(path)

    @classmethod
    def find(cls):
        result = {}

        for id in cls.PATHS.keys():
            path = _presolver.resolve(cls.path(id))
            result[id] = path.is_file()

        return result
