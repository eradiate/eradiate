"""Data handling facilities."""

import functools
import os

import xarray as xr

from ..util.presolver import PathResolver

presolver = PathResolver()


@functools.lru_cache(maxsize=32)
def get(path):
    """Return a data set based on queried path. Results produce by this function
    are cached using an
    `LRU <https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)>`_
    policy to minimise hard drive access.

    Parameter ``path`` (path-like):
        Path to the requested resource, resolved by the :class:`.FileResolver`.
    """

    fname = presolver.resolve(path)
    ext = os.path.splitext(fname)[1]

    if ext == ".nc":
        return xr.load_dataset(fname)

    raise ValueError(f"cannot load resource {fname}")


SOLAR_IRRADIANCE_SPECTRA = {
    "blackbody_sun": "spectra/blackbody_sun.nc",
    "thuillier_2003": "spectra/thuillier_2003.nc",
}
"""This dictionary provides access to solar irradiance data sets shipped with 
Eradiate. Keys are unique identifiers associated with shipped data sets, and
values are relative paths to the corresponding netCDF file in the Eradiate data
directory. These relative paths can be used with :func:`get` to conveniently
access the data sets.

.. admonition:: Example

   To access the Thuillier irradiance spectrum :cite:`Thuillier2003SolarSpectralIrradiance`,
   the following can be done:
   
   .. code:: python
   
      import eradiate.data as data
      
      ds = data.get(data.SOLAR_IRRADIANCE_SPECTRA["thuillier_2003"])

The following table lists available data sets and their corresponding 
identifiers.

.. list-table::
   :widths: 1 1 1
   :header-rows: 1
   
   * - Key
     - Reference
     - Spectral range [nm]
   * - ``blackbody_sun``
     - :cite:`Liou2002IntroductionAtmosphericRadiation`
     - [280, 2400]
   * - ``thuillier_2003``
     - :cite:`Thuillier2003SolarSpectralIrradiance`
     - [200, 2397]
"""
