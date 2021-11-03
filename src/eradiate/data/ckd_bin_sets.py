"""
CKD spectral bin definitions.
"""

import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver

r"""
CKD bin set file format
=======================

CKD bin sets definitions are provided as NetCDF files.

Coordinates (\* means also dimension):

* \* ``bin``: bin identifier (str)

Variables:

* ``wmin``[``bin``]: lower bound wavelength in nm (float)
* ``wmax``[``bin``]: upper bound wavelength in nm (float)

Metadata:

* ``quadrature_type``: quadrature type (str)
* ``quadrature_n``: number of quadrature points (int)
"""


class _CKDBinSetGetter(DataGetter):
    PATHS = {
        "10nm": "ckd/bin_sets/10nm.nc",
        "10nm_test": "ckd/bin_sets/10nm_test.nc",
    }

    @classmethod
    def open(cls, id):
        # Single-file version
        ds = xr.open_dataset(_presolver.resolve(cls.PATHS[id]))
        return ds

    @classmethod
    def find(cls):
        result = {}

        for id in cls.PATHS.keys():
            paths = _presolver.glob(cls.path(id))
            result[id] = bool(len(list(paths)))

        return result
