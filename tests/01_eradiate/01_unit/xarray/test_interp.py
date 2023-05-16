import numpy as np
import xarray as xr

from eradiate.xarray.interp import dataarray_to_rgb, film_to_angular


def test_film_to_angular():
    # This does not test the output very well, but it will fail if any
    # subcomponent is broken.
    # Actual behaviour is better tested manually by comparing film-space and
    # polar plots.

    # Create test data array
    da = xr.DataArray(
        [[i + j for i in range(16)] for j in range(32)],
        coords=(
            ("x", np.arange(0.5, 32, 1) / 32),
            ("y", np.arange(0.5, 16, 1) / 16),
        ),
    )

    theta = np.radians(np.arange(0.0, 85.0, 5))
    phi = np.radians(np.arange(0.0, 360.01, 5))

    da_angular = film_to_angular(da, theta=theta, phi=phi)
    assert set(da_angular.coords) == {"theta", "phi"}


def test_dataarray_to_rgb():
    r = 0.25 * np.ones((3, 2, 1))
    g = 0.50 * np.ones((3, 2, 1))
    b = 0.75 * np.ones((3, 2, 1))
    da = xr.DataArray(
        data=np.concatenate([b, g, r], axis=2),
        dims=["x", "y", "w"],
        coords={"x": [0, 1, 2], "y": [0, 1], "w": [440, 550, 660]},
    )

    channels = [("w", 660), ("w", 550), ("w", 440)]

    # Everything off: the data are just assembled in the same order as the channels
    img = dataarray_to_rgb(da, channels, normalize=False, gamma_correction=False)
    assert isinstance(img, np.ndarray)
    assert np.allclose(img, da.values[:, :, ::-1])

    # Normalisation scales the data
    img = dataarray_to_rgb(da, channels, normalize=True, gamma_correction=False)
    assert np.max(img) == 1.0

    # Gamma correction applies the gamma 2.2 operator
    img = dataarray_to_rgb(da, channels, normalize=False, gamma_correction=True)
    assert np.allclose(img, da.values[:, :, ::-1] ** (1.0 / 2.2))

    # By default, both are applied
    img = dataarray_to_rgb(da, channels)
    assert np.allclose(img, (da.values[:, :, ::-1] / float(da.max())) ** (1.0 / 2.2))
