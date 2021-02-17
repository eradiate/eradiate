"""Spectral response function data sets shipped with Eradiate.

**Category ID**: ``spectral_response_function``

.. list-table:: Available data sets and corresponding identifiers
   :widths: 1 1 1
   :header-rows: 1

   * - Data set ID
     - Reference
     - Spectral range [nm]
   * - ``sentinel-3a-slstr-s1``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [515, 595]
   * - ``sentinel-3a-slstr-s2``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [619, 699]
   * - ``sentinel-3a-slstr-s3``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [825, 905]
   * - ``sentinel-3a-slstr-s4``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [1345, 1405]
   * - ``sentinel-3a-slstr-s5``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [1490, 1730]
   * - ``sentinel-3a-slstr-s6``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [2150, 2350]
   * - ``sentinel-3a-slstr-s7``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [2980, 4500]
   * - ``sentinel-3a-slstr-s8``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [9050, 12650]
   * - ``sentinel-3a-slstr-s9``
     - :cite:`RAL2015Sentinel3ASLSTR`
     - [10000, 14000]
   * - ``sentinel-3b-slstr-s1``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [515, 595]
   * - ``sentinel-3b-slstr-s2``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [619, 699]
   * - ``sentinel-3b-slstr-s3``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [825, 905]
   * - ``sentinel-3b-slstr-s4``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [1345, 1405]
   * - ``sentinel-3b-slstr-s5``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [1490, 1730]
   * - ``sentinel-3b-slstr-s6``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [2150, 2350]
   * - ``sentinel-3b-slstr-s7``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [2980, 4500]
   * - ``sentinel-3b-slstr-s8``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [9050, 12650]
   * - ``sentinel-3b-slstr-s9``
     - :cite:`RAL2017Sentinel3BSLSTR`
     - [10000, 14000]


.. admonition:: Spectral response functions dataset specification

   The data structure is a :class:`~xarray.Dataset` with specific data variables, dimensions and data coordinates.

   Data variables must be:

   - ``srf``: spectral response values [dimensionless],

    The `array dimension <http://xarray.pydata.org/en/stable/terminology.html>`_ of ``srf`` is ``w``.

   The data coordinate is:

   - ``w``: wavelength values [nm].
"""

import xarray as xr

from .core import DataGetter
from .. import path_resolver as _presolver


class _SpectralResponseFunctionGetter(DataGetter):
    PATHS = {
        f"sentinel-3{satellite}-slstr-s{channel}":
        f"spectra/srf/sentinel-3{satellite}-slstr-s{channel}.nc"
        for satellite in ["a", "b"] for channel in range(1, 10)
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
