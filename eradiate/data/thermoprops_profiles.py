"""
Atmospheric thermophysical properties profiles data sets shipped with Eradiate.
"""
import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver


class _ThermoPropsProfilesGetter(DataGetter):
    PATHS = {
        "afgl1986-midlatitude_summer": "thermoprops_profiles/afgl1986-midlatitude_summer.nc",
        "afgl1986-midlatitude_winter": "thermoprops_profiles/afgl1986-midlatitude_winter.nc",
        "afgl1986-subarctic_summer": "thermoprops_profiles/afgl1986-subarctic_summer.nc",
        "afgl1986-subarctic_winter": "thermoprops_profiles/afgl1986-subarctic_winter.nc",
        "afgl1986-tropical": "thermoprops_profiles/afgl1986-tropical.nc",
        "afgl1986-us_standard": "thermoprops_profiles/afgl1986-us_standard.nc",
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
