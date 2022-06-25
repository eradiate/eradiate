"""
Manage Eradiate's data files.
"""

import sys

from ..typing import PathLike
from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_store": ["data_store", "init_data_store"],
        "_core": ["DataStore"],
        "_blind_directory": ["BlindDirectoryDataStore"],
        "_blind_online": ["BlindOnlineDataStore"],
        "_safe_directory": ["SafeDirectoryDataStore"],
        "_safe_online": ["SafeOnlineDataStore"],
        "_multi": ["MultiDataStore"],
    },
)
xr = lazy_loader.load("xarray")

del lazy_loader


# -- Access functions ----------------------------------------------------------


def open_dataset(filename: PathLike, **kwargs) -> "xarray.Dataset":
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


def load_dataset(*args, **kwargs) -> "xarray.Dataset":
    """
    Open, load into memory, and close a dataset from the online data store
    (requires Internet access).

    See Also
    --------
    open_dataset
    """
    with open_dataset(*args, **kwargs) as ds:
        return ds.load()


__all__.extend(["open_dataset", "load_dataset"])
