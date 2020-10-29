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
    _PATHS = {
        "blackbody_sun": "spectra/blackbody_sun.nc",
        "meftah_2017": "spectra/meftah_2017.nc",
        "thuillier_2003": "spectra/thuillier_2003.nc",
        "whi_2008": "spectra/whi_2008_time_period_1.nc",  # alias
        "whi_2008_1": "spectra/whi_2008_time_period_1.nc",
        "whi_2008_2": "spectra/whi_2008_time_period_2.nc",
        "whi_2008_3": "spectra/whi_2008_time_period_3.nc",
    }

    @classmethod
    def open(cls, id):
        path = _presolver.resolve(cls.path(id))
        return xr.open_dataset(path)
