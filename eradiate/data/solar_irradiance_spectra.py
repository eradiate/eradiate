"""Solar irradiance spectrum data sets shipped with Eradiate.
"""

import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver


class _SolarIrradianceGetter(DataGetter):
    PATHS = {
        "blackbody_sun": "spectra/solar_irradiance/blackbody_sun.nc",
        "meftah_2017": "spectra/solar_irradiance/meftah_2017.nc",
        "thuillier_2003": "spectra/solar_irradiance/thuillier_2003.nc",
        "solid_2017_mean": "spectra/solar_irradiance/solid_2017_mean.nc",
        "solid_2017": "spectra/solar_irradiance/solid_2017.nc",
        "whi_2008": "spectra/solar_irradiance/whi_2008_time_period_1.nc",  # alias
        "whi_2008_1": "spectra/solar_irradiance/whi_2008_time_period_1.nc",
        "whi_2008_2": "spectra/solar_irradiance/whi_2008_time_period_2.nc",
        "whi_2008_3": "spectra/solar_irradiance/whi_2008_time_period_3.nc",
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
