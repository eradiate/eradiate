"""Data handling facilities."""

import functools
import os

import xarray as xr

from ..util.presolver import PathResolver

presolver = PathResolver()


@functools.lru_cache(maxsize=32)
def get(path):
    """Return a data set based on queried path. Results produce by this function
    are cached using an
    `LRU <https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)>`_
    policy to minimise hard drive access.

    Parameter ``path`` (path-like):
        Path to the requested resource, resolved by the :class:`.FileResolver`.
    """

    fname = presolver.resolve(path)
    ext = os.path.splitext(fname)[1]

    if ext == ".nc":
        return xr.load_dataset(fname)

    raise ValueError(f"cannot load resource {fname}")
