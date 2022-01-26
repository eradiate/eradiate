"""
Volume data file I/O helpers.
"""

import os
import typing as t

import numpy as np
import xarray as xr

from ..typing import PathLike


def write_binary_grid3d(
    filename: PathLike, values: t.Union[np.ndarray, xr.DataArray]
) -> None:
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
    from mitsuba.render import VolumeGrid

    if isinstance(values, xr.DataArray):
        values = values.values

    if not isinstance(values, np.ndarray):
        raise TypeError(
            f"unsupported data type {type(values)} "
            f"(expected numpy array or xarray DataArray)"
        )

    if values.ndim not in {3, 4}:
        raise ValueError(
            f"'values' must have 3 or 4 dimensions " f"(got shape {values.shape})"
        )

    VolumeGrid(values.astype(np.float32)).write(str(filename))


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
    from mitsuba.render import VolumeGrid

    return np.array(VolumeGrid(str(filename)))
