"""
Microphysical properties data sets.
"""

import pandas as pd
import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver


class _Microprops(DataGetter):
    PATHS = {
        "shettle_fenn_1979_table_2": "microprops/shettle_fenn_1979_table_2.nc",
        "shettle_fenn_1979_table_3": "microprops/shettle_fenn_1979_table_3.nc",
        "shettle_fenn_1979_table_4a": "microprops/shettle_fenn_1979_table_4a.nc",
        "shettle_fenn_1979_table_4b": "microprops/shettle_fenn_1979_table_4b.nc",
    }

    @classmethod
    def open(cls, id):
        path = path_resolver.resolve(cls.path(id))
        return xr.open_dataset(path)
