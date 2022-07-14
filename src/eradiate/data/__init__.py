"""
Manage Eradiate's data files.
"""

__all__ = [
    "data_store",
    "init_data_store",
    "open_dataset",
    "load_dataset",
    "DataStore",
    "BlindOnlineDataStore",
    "MultiDataStore",
    "SafeDirectoryDataStore",
    "SafeOnlineDataStore",
]

import sys

import xarray as xr

from ._blind_online import BlindOnlineDataStore
from ._core import DataStore
from ._multi import MultiDataStore
from ._safe_directory import SafeDirectoryDataStore
from ._safe_online import SafeOnlineDataStore
from ._store import init_data_store
from ..typing import PathLike

# -- Data store ----------------------------------------------------------------


def __getattr__(name):
    if name == "data_store":
        from ._store import data_store

        return data_store

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# -- Access functions ----------------------------------------------------------


def open_dataset(filename: PathLike, **kwargs) -> xr.Dataset:
    """
    Open a dataset from the online data store (requires Internet access).

    Parameters
    ----------
    filename : path-like
        Path to the requested dataset, relative to the data store root.

    **kwargs
        Forwarded to :func:`xarray.open_dataset`.

    Returns
    -------
    Dataset
        Open dataset, if found in the data store.

    See Also
    --------
    xarray.open_dataset
    """
    data_store = getattr(sys.modules[__name__], "data_store")
    filename = data_store.fetch(filename)
    return xr.open_dataset(filename, **kwargs)


def load_dataset(*args, **kwargs) -> xr.Dataset:
    """
    Open, load into memory, and close a dataset from the online data store
    (requires Internet access).

    See Also
    --------
    open_dataset
    """
    with open_dataset(*args, **kwargs) as ds:
        return ds.load()
