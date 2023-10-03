from __future__ import annotations

import warnings

import mitsuba as mi
import numpy as np
import xarray as xr


def bitmap_to_dataarray(bmp: mi.Bitmap, dtype="float64") -> xr.DataArray:
    """
    Format Mitsuba bitmap data as an xarray data array.

    Parameters
    ----------
    bmp : mitsuba.core.Bitmap or ndarray
        Mitsuba bitmap to be converted to a dataset. A Numpy array can also be
        passed directly for compatibility (this feature is deprecated).

    dtype : dtype, optional
        Data type, forwarded to :func:`numpy.array`.

    Returns
    -------
    da : DataArray
        Bitmap data as an xarray array.

    Raises
    ------
    ValueError
        If `bmp` has an unsupported pixel format.
    """
    img = np.array(bmp, dtype=dtype)

    if isinstance(bmp, mi.Bitmap):
        try:
            pixel_formats = {
                mi.Bitmap.PixelFormat.Y: ["Y"],
                mi.Bitmap.PixelFormat.YA: ["Y", "A"],
                mi.Bitmap.PixelFormat.RGB: ["R", "G", "B"],
                mi.Bitmap.PixelFormat.RGBA: ["R", "G", "B", "A"],
                mi.Bitmap.PixelFormat.XYZ: ["X", "Y", "Z"],
                mi.Bitmap.PixelFormat.XYZA: ["X", "Y", "Z", "A"],
            }
            channels = pixel_formats[bmp.pixel_format()]
        except KeyError:
            raise ValueError(f"unsupported bitmap pixel format {bmp.pixel_format()}")

    else:
        warnings.warn(
            "Passing an array is deprecated (coordinate detection is limited)",
            DeprecationWarning,
        )
        if len(img.shape) < 3:
            channels = ["Y"]
        else:
            channels = {1: ["Y"], 3: ["R", "G", "B"]}[img.shape[2]]

    height = img.shape[0]
    width = img.shape[1]

    result = xr.DataArray(
        data=np.reshape(img, (height, width, -1)),
        dims=["y_index", "x_index", "channel"],
        coords={
            "y_index": (
                "y_index",
                range(height),
                {"long_name": "height pixel index"},
            ),
            "y": (
                "y_index",
                np.linspace(0, 1, height),
                {"long_name": "film height coordinate"},
            ),
            "x_index": (
                "x_index",
                range(width),
                {"long_name": "width pixel index"},
            ),
            "x": (
                "x_index",
                np.linspace(0, 1, width),
                {"long_name": "film width coordinate"},
            ),
            "channel": ("channel", channels, {"long_name": "film spectral channel"}),
        },
    )

    return result


def bitmap_to_dataset(bmp: mi.Bitmap, dtype="float64") -> xr.Dataset:
    """
    Format Mitsuba bitmap data as an xarray dataset.

    Parameters
    ----------
    bmp : mitsuba.core.Bitmap or ndarray
        Mitsuba bitmap to be converted to a dataset. A Numpy array can also be
        passed directly for compatibility (this feature is deprecated).

    dtype : dtype, optional
        Data type, forwarded to :func:`numpy.array`.

    Returns
    -------
    dataset : Dataset
        Bitmap data as an xarray dataset.

    See Also
    --------
    :func:`bitmap_to_dataarray`

    Notes
    -----
    This function exists for backward compatibility purposes.
    It calls :func:`bitmap_to_dataarray` and forwards its arguments to it, then
    wraps the generated data array in a dataset with an ``img`` data variable.
    """

    da = bitmap_to_dataarray(bmp, dtype)
    return xr.Dataset(data_vars={"img": da})
