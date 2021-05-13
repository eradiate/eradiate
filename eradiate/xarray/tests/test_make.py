import numpy as np
import xarray as xr

from eradiate.xarray.make import film_to_angular, make_dataarray
from eradiate.xarray.metadata import VarSpec


def test_make_dataarray():
    # Check that basic array creation works
    da = make_dataarray(
        data=np.ones((1, 1, 1, 1, 1)),
        coords={
            "vza": [0.0],
            "vaa": [0.0],
            "sza": [0.0],
            "saa": [0.0],
            "w": [500.0],
        },
        dims=("vza", "vaa", "sza", "saa", "w"),
        var_spec=VarSpec(coord_specs="angular_observation"),
    )

    assert da.dims == ("vza", "vaa", "sza", "saa", "w")
    assert set(da.coords.keys()) == {"vza", "vaa", "sza", "saa", "w"}

    # Check that metadata are applied properly if required
    da = make_dataarray(
        data=np.ones((1, 1, 1, 1, 1)),
        coords={
            "vza": [0.0],
            "vaa": [0.0],
            "sza": [0.0],
            "saa": [0.0],
            "w": [500.0],
        },
        dims=("vza", "vaa", "sza", "saa", "w"),
        var_spec=VarSpec(coord_specs="angular_observation"),
    )

    assert da.coords["vza"].attrs == {
        "standard_name": "viewing_zenith_angle",
        "units": "deg",
        "long_name": "viewing zenith angle",
    }


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
