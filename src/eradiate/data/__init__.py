"""
Manage Eradiate's data files.
"""

import sys

from ..typing import PathLike
from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
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
