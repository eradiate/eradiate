"""
Volume data file I/O helpers.
"""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import xarray as xr

from ..typing import PathLike


def write_binary_grid3d(filename: PathLike, values: np.ndarray | xr.DataArray) -> None:
    """
    Write volume data to a binary file so that a ``gridvolume`` kernel plugin
    can be instantiated with that file.

    Parameters
    ----------
    filename : path-like
        File name.

    values : ndarray or DataArray
        Data array to output to the volume data file. This array must have 3 or
        4 dimensions (x, y, z, spectrum). If the array is 3-dimensional, it will
        automatically be assumed to have only one spectral channel.
    """
    if isinstance(values, xr.DataArray):
        values = values.values

    if not isinstance(values, np.ndarray):
        raise TypeError(
            f"unsupported data type {type(values)} "
            f"(expected numpy array or xarray DataArray)"
        )

    if values.ndim not in {3, 4}:
        raise ValueError(
            f"'values' must have 3 or 4 dimensions (got shape {values.shape})"
        )

    mi.VolumeGrid(values.astype(np.float32)).write(str(filename))


def read_binary_grid3d(filename: PathLike) -> np.ndarray:
    """
    Reads a volume data binary file.

    Parameters
    ----------
    filename : path-like
        File name.

    Returns
    -------
    ndarray
        Values.
    """
    return np.array(mi.VolumeGrid(str(filename)))
