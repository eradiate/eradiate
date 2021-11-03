import warnings

import numpy as np
import xarray as xr


def bitmap_to_dataset(bmp: "mitsuba.core.Bitmap", dtype=float) -> xr.Dataset:
    """
    Format Mitsuba bitmap data as an xarray dataset.

    Parameters
    ----------
    bmp : mitsuba.core.Bitmap or ndarray
        Mitsuba bitmap to be converted to a dataset. A Numpy array can also be
        passed directly for compatibility (this feature is deprecated).

    dtype : dtype
        Data type, forwarded to :func:`numpy.array`.

    Returns
    -------
    dataset : Dataset
        Bitmap data as an xarray dataset.

    Raises
    ------
    ValueError
        If `bmp` has an unsupported pixel format.
    """
    from mitsuba.core import Bitmap

    img = np.array(bmp, dtype=dtype)

    try:
        channels = {
            Bitmap.PixelFormat.Y: ["Y"],
            Bitmap.PixelFormat.YA: ["Y", "A"],
            Bitmap.PixelFormat.RGB: ["R", "G", "B"],
            Bitmap.PixelFormat.RGBA: ["R", "G", "B", "A"],
            Bitmap.PixelFormat.XYZ: ["X", "Y", "Z"],
            Bitmap.PixelFormat.XYZA: ["X", "Y", "Z", "A"],
            Bitmap.PixelFormat.XYZAW: ["X", "Y", "Z", "A", "W"],
        }[bmp.pixel_format()]

    except KeyError:
        raise ValueError(f"unsupported bitmap pixel format {bmp.pixel_format()}")

    except AttributeError:
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

    result = xr.Dataset(
        data_vars={"img": (["y_index", "x_index", "channel"], img)},
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
