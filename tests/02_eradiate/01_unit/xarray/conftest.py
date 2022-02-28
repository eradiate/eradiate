import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def dataarray_without_metadata():
    return xr.DataArray(
        data=np.random.random((1, 1, 1, 1, 1)),
        dims=["vza", "vaa", "sza", "saa", "w"],
        coords={
            "vza": [0.0],
            "vaa": [0.0],
            "sza": [0.0],
            "saa": [0.0],
            "w": [500.0],
        },
    )


@pytest.fixture
def dataset_without_metadata():
    return xr.Dataset(
        data_vars={
            "p": (("z_layer", "z_level", "species"), np.random.random((1, 1, 1))),
        },
        coords={
            "z_layer": ("z_layer", [0.0]),
            "z_level": ("z_level", [0.0]),
            "species": ("species", ["foo"]),
        },
    )
