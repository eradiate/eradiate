from __future__ import annotations

from contextlib import contextmanager

import xarray as xr

from ..typing import PathLike


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
    from ._store import data_store

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


@contextmanager
def open_datasets(paths):
    try:
        datasets = [open_dataset(path) for path in paths]
        yield datasets
    finally:
        for ds in datasets:
            ds.close()
