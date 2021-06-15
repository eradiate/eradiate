import numpy as np
import pytest
import xarray as xr

from eradiate.xarray.math import ratios


@pytest.fixture
def random_arrays():
    da1 = xr.DataArray(
        np.random.random((2, 5)),
        dims=["w", "z"],
        coords={"w": [500.0, 600.0], "z": np.linspace(0, 80, 5)},
    )

    da2 = xr.DataArray(
        np.random.random((2, 5)),
        dims=["w", "z"],
        coords={"w": [500.0, 600.0], "z": np.linspace(0, 80, 5)},
    )
    return [da1, da2]


def test_ratios(random_arrays):
    """Returns a DataArray."""
    r = ratios(random_arrays)
    assert isinstance(r, xr.DataArray)


def test_ratios_dim_coord_input(random_arrays):
    """Returned DataArray has the same dimension coordinates as input arrays."""
    r = ratios(random_arrays)
    assert all([c in r.coords for c in random_arrays[0].coords])
    assert all([d in r.dims for d in random_arrays[0].dims])


def test_ratios_dim_coord_i(random_arrays):
    """Returned DataArray has a 'i' dimension coordinate."""
    r = ratios(random_arrays)
    assert "i" in r.dims and "i" in r.coords


@pytest.fixture
def ones_arrays():
    da1 = xr.DataArray(
        np.ones((2, 5)),
        dims=["w", "z"],
        coords={"w": [500.0, 600.0], "z": np.linspace(0, 80, 5)},
    )

    da2 = xr.DataArray(
        np.ones((2, 5)),
        dims=["w", "z"],
        coords={"w": [500.0, 600.0], "z": np.linspace(0, 80, 5)},
    )
    return [da1, da2]


def test_ratios_values(ones_arrays):
    """Computed ratios have values = 50% when tow input arrays are 1-like."""
    r = ratios(ones_arrays)
    assert np.allclose(r.values, 0.5)
