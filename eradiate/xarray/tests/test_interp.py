import numpy as np
import xarray as xr

from eradiate.xarray.interp import film_to_angular


def test_film_to_angular():
    # This does not test the output very well, but it will fail if any
    # sub-component is broken.
    # Actual behaviour is better tested manually by comparing pixel-space and
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
