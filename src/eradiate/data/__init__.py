"""
Manage Eradiate's data files.
"""

__all__ = [
    "data_store",
    "open_dataset",
    "load_dataset",
    "DataStore",
    "DirectoryDataStore",
    "OnlineDataStore",
    "MultiDataStore",
]

import xarray as xr

from ._core import DataStore
from ._directory import DirectoryDataStore
from ._multi import MultiDataStore
from ._online import OnlineDataStore
from .._config import config as _config
from ..typing import PathLike

#: Global data store.
data_store = MultiDataStore(
    stores=[
        DirectoryDataStore(path=_config.dir / "resources/data"),
        OnlineDataStore(
            base_url="http://eradiate.eu/data/store",
            path=_config.download_dir,
        ),
    ]
)


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
