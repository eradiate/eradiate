import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from eradiate.frame import angles_to_direction
from eradiate.warp import uniform_hemisphere_to_square


def make_dataarray(data, coords=None, dims=None, var_spec=None):
    """Create a :class:`~xarray.DataArray` with default metadata.

    Parameter ``data``:
        Data forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``coords``:
        Coordinates forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``dims``:
        Dimension names forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``var_spec`` (:class:`.VarSpec` or None):
        If not ``None``, data variable specification used to apply default
        metadata.
    """

    dataarray = xr.DataArray(data, coords=coords, dims=dims)
    if var_spec is not None:
        dataarray.ert.normalize_metadata(var_spec)

    return dataarray


def film_to_angular(
    da: xr.DataArray,
    theta: ArrayLike,
    phi: ArrayLike,
    x_label: str = "x",
    y_label: str = "y",
    theta_label: str = "theta",
    phi_label: str = "phi",
):
    """
    Interpolate a hemispherical film data array on an angular grid.

    Parameter ``da`` (:class:`xarray.DataArray`):
        Data array with film coordinate pixels to interpolate on an angular grid.

    Parameter ``theta`` (array-like):
        List of zenith angle values.

    Parameter ``phi`` (array-like):
        List of azimuth angle values.

    Parameter ``x_label`` (str):
        Label for the width pixel coordinate.

    Parameter ``y_label`` (str):
        Label for the height pixel coordinate.

    Parameter ``theta_label`` (str):
        Label for the zenith angle coordinate.

    Parameter ``phi_label`` (str):
        Label for the azimuth angle coordinate.

    Returns → :class:`xarray.DataArray`:
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
