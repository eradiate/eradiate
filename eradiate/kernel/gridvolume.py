import pathlib
import struct
import typing as t

import numpy as np
import xarray as xr


def write_binary_grid3d(
    filename: t.Union[str, pathlib.Path], values: t.Union[np.ndarray, xr.DataArray]
) -> None:
    """
    Write volume data to a binary file so that a ``gridvolume`` kernel plugin
    can be instantiated with that file.

    Parameters
    ----------
    filename : path-like
        File name.

    Parameters
    ----------
    values : ndarray or DataArray
        Data array to output to the volume data file. This array must 3 or 4
        dimensions (x, y, z, spectrum). If the array is 3-dimensional, it will
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
            f"'values' must have 3 or 4 dimensions " f"(got shape {values.shape})"
        )

    # note: this is an exact copy of the function write_binary_grid3d from
    # https://github.com/mitsuba-renderer/mitsuba-data/blob/master/tests/scenes/participating_media/create_volume_data.py

    with open(filename, "wb") as f:
        f.write(b"V")
        f.write(b"O")
        f.write(b"L")
        f.write(np.uint8(3).tobytes())  # Version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(values.shape[0]).tobytes())  # size
        f.write(np.int32(values.shape[1]).tobytes())
        f.write(np.int32(values.shape[2]).tobytes())
        if values.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(values.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(values.ravel().astype(np.float32).tobytes())


def read_binary_grid3d(filename: str) -> np.ndarray:
    """
    Reads a volume data binary file.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    ndarray
        Values.
    """

    with open(filename, "rb") as f:
        file_content = f.read()
        _shape = struct.unpack("iii", file_content[8:20])  # shape of the values array
        _num = np.prod(np.array(_shape))  # number of values
        values = np.array(struct.unpack("f" * _num, file_content[48:]))
        # file_type = struct.unpack("ccc", file_content[:3]),
        # version = struct.unpack("B", file_content[3:4]),
        # type = struct.unpack("i", file_content[4:8]),
        # channels = struct.unpack("i", file_content[20:24]),
        # bbox = struct.unpack("ffffff", file_content[24:48]),

    return values
