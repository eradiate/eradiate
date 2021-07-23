"""Chemistry data sets
"""

import xarray as xr

from .core import DataGetter
from .._presolver import path_resolver as _presolver


class _ChemistryGetter(DataGetter):
    PATHS = {"molecular_masses": "chemistry/molecular_masses.nc"}

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
