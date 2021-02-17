import numpy as np
import xarray as xr

from eradiate.xarray.select import (
    plane,
    pplane
)


def test_plane():
    """Test the plane method by creating a plane view from a DataArray and
    comparing it to a reference."""

    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    arr = xr.DataArray(
        data,
        dims=["theta_o", "phi_o"],
        coords={"theta_o": [0, 30, 60, 90], "phi_o": [0, 90, 180, 270]},
    )

    p = plane(arr, phi=90)
    assert np.all(p["theta_o"] == [-90, -60, -30, 0, 30, 60, 90])
    assert np.all(p.values.squeeze() == [16, 12, 8, 2, 6, 10, 14])

    arr = xr.DataArray(
        data,
        dims=["theta", "phi"],
        coords={"theta": [0, 30, 60, 90], "phi": [0, 90, 180, 270]},
    )

    p = plane(arr, phi=90, theta_dim="theta", phi_dim="phi")
    assert np.all(p["theta"] == [-90, -60, -30, 0, 30, 60, 90])
    assert np.all(p.values.squeeze() == [16, 12, 8, 2, 6, 10, 14])


def test_pplane():
    """Test the pplane convenience function by comparing it to the result of
    manually calling plane with the corresponding arguments."""

    data = np.random.rand(4, 4, 4, 4)

    arr = xr.DataArray(
        data,
        dims=["vza", "vaa", "sza", "saa"],
        coords={
            "vza": [0, 30, 60, 90],
            "vaa": [0, 90, 180, 270],
            "sza": [0, 30, 60, 90],
            "saa": [0, 90, 180, 270],
        },
    )
    pp = pplane(arr, sza=60, saa=90)
    p = plane(arr.sel(sza=60, saa=90), phi=90, theta_dim="vza", phi_dim="vaa")

    assert np.all(pp == p)
