"""Solar irradiance spectrum data sets shipped with Eradiate.

**Category ID**: ``solar_irradiance_spectrum``

.. list-table:: Available data sets and corresponding identifiers
   :widths: 1 1 1
   :header-rows: 1

   * - Data set ID
     - Reference
     - Spectral range [nm]
   * - ``blackbody_sun``
     - :cite:`Liou2002IntroductionAtmosphericRadiation`
     - [280, 2400]
   * - ``meftah_2017``
     - :cite:`Meftah2017SOLARISSReference`
     - [165, 3000.1]
   * - ``solid_2017_mean``
     - :cite:`Haberreiter2017ObservationalSolarIrradiance`
     - [0.5, 1991.5]
   * - ``thuillier_2003``
     - :cite:`Thuillier2003SolarSpectralIrradiance`
     - [200, 2397]
   * - ``whi_2008``
     - :cite:`Woods2008SolarIrradianceReference`
     - [116.05, 2399.95]
   * - ``whi_2008_1``
     - :cite:`Woods2008SolarIrradianceReference`
     - [116.05, 2399.95]
   * - ``whi_2008_2``
     - :cite:`Woods2008SolarIrradianceReference`
     - [116.05, 2399.95]
   * - ``whi_2008_3``
     - :cite:`Woods2008SolarIrradianceReference`
     - [116.05, 2399.95]
"""

import xarray as xr

from .core import DataGetter
from ..util.presolver import PathResolver

_presolver = PathResolver()


class _SolarIrradianceGetter(DataGetter):
    PATHS = {
        "blackbody_sun": "spectra/solar_irradiance/blackbody_sun.nc",
        "meftah_2017": "spectra/solar_irradiance/meftah_2017.nc",
        "thuillier_2003": "spectra/solar_irradiance/thuillier_2003.nc",
        "solid_2017_mean": "spectra/solar_irradiance/solid_2017_mean.nc",
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
