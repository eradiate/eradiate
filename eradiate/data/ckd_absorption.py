"""
CKD absorption data.
"""

import pandas as pd
import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver

r"""
CKD data file format
====================

CKD absorption data are provided as single NetCDF files.

.. note::
   An integer 'bd' dimension unrolls the (bin, index) multi-index. This could be
   improved by splitting the bindex into 2 dimensions, possibly leaving
   undefined values at indexes where no quadrature point is defined for the
   considered CKD bin. How that would play with multi-index definition, however,
   is yet to be investigated.
   
Coordinates (\* means also dimension):

* \* ``z``: altitude
* `Multi-index <http://xarray.pydata.org/en/stable/user-guide/data-structures.html?highlight=multiindex#multiindex-coordinates>`_ 
  \* ``bd``

  * ``bin``: bin ID (str)
  * ``index``: index value within bin, *i.e.* quadrature node index (int)

Variables:

* ``k`` [``z``, ``bd``]: absorption coefficient in m^-1 (float)

Metadata:

* ``bin_set``: ID of the associated bin set
"""


class _CKDAbsorptionGetter(DataGetter):
    PATHS = {
        "afgl_1986-us_standard-10nm_test": "ckd/absorption/10nm_test/afgl_1986-us_standard-10nm_test.nc",
    }

    @classmethod
    def open(cls, id):
        # Multi-file version (disabled for now)
        # glob_paths = sorted(list(_presolver.glob(cls.PATHS[id])))
        # ds = xr.open_mfdataset(glob_paths, combine="nested", concat_dim="bd").compute()

        # Single-file version
        ds = xr.open_dataset(_presolver.resolve(cls.PATHS[id]))

        # Combine the 'bin' and 'index' coordinates into a multi-index, then reindex dataset
        idx = pd.MultiIndex.from_arrays(
            (ds.bin.values, ds.index.values), names=("bin", "index")
        )
        ds = ds.drop_vars(("bin", "index"))
        ds = ds.reindex({"bd": idx})

        return ds

    @classmethod
    def find(cls):
        result = {}

        for id in cls.PATHS.keys():
            paths = _presolver.glob(cls.path(id))
            result[id] = bool(len(list(paths)))

        return result
