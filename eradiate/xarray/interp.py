__all__ = ["film_to_angular"]

import numpy as np
import xarray as xr

from ..frame import angles_to_direction
from ..warp import uniform_hemisphere_to_square


def film_to_angular(
    da: xr.DataArray,
    theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
    x_label: str = "x",
    y_label: str = "y",
    theta_label: str = "theta",
    phi_label: str = "phi",
) -> xr.DataArray:
    """
    Interpolate a hemispherical film data array on an angular grid.

    Parameters
    ----------
    da : DataArray
        Data array with film coordinate pixels to interpolate on an angular grid.

    theta : array-like
        List of zenith angle values.

    phi : array-like
        List of azimuth angle values.

    x_label : str, default: "x"
        Label for the width pixel coordinate.

    y_label : str, default: "y"
        Label for the height pixel coordinate.

    theta_label : str, default: "theta"
        Label for the zenith angle coordinate.

    phi_label : str, default: "phi"
        Label for the azimuth angle coordinate.

    Returns
    -------
    DataArray
        Data array interpolated on the specified angular grid.
    """

    # Interpolate values on angular grid
    data = np.empty((len(phi), len(theta)))

    # Map angular grid points to (x, y) space
    for i, ph in enumerate(phi):
        xs = np.empty_like(theta)
        ys = np.empty_like(theta)

        angles = np.array([[th, ph] for th in theta.ravel()])
        directions = angles_to_direction(angles)
        film_coords = uniform_hemisphere_to_square(directions)
        xs.ravel()[:] = film_coords[:, 0]
        ys.ravel()[:] = film_coords[:, 1]

        x = xr.DataArray(xs, dims=theta_label)
        y = xr.DataArray(ys, dims=theta_label)
        data[i, :] = da.interp(**{x_label: x, y_label: y}).values

    return xr.DataArray(
        data,
        coords=(
            (phi_label, phi),
            (theta_label, theta),
        ),
        dims=(phi_label, theta_label),
    )
