"""
Microphysical properties data sets.
"""

import pandas as pd
import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver


class _Microprops(DataGetter):
    PATHS = {
        "shettle_fenn_1979_table_2": "microprops/shettle_fenn_1979_table_2.csv",
        "shettle_fenn_1979_table_3": "microprops/shettle_fenn_1979_table3.csv",
    }

    @classmethod
    def open(cls, id):
        path = path_resolver.resolve(cls.path(id))
        return pd.read_csv(path)
