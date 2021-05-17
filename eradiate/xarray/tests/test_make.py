import numpy as np

from eradiate.xarray.make import make_dataarray
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
