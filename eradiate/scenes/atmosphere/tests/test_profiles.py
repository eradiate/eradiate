import pytest
import xarray as xr
from eradiate.scenes.atmosphere.profiles import *


def test_check():
    # profile with a missing data variable (total number density)
    p = xr.Dataset(
        data_vars={
            "pressure": ("altitude", np.random.rand(100), {"units": "Pa"}),
            "temperature": ("altitude", np.random.rand(100), {"units": "K"}),
            "number_density": (("species", "altitude"), np.random.rand(2, 100), {"units": "m^-3"}),
            "upper_bound": ("altitude", np.random.rand(100), {"units": "m"}),
            "lower_bound": ("altitude", np.random.rand(100), {"units": "m"})},
        coords={
            "altitude": ("altitude", np.linspace(0, 86e3, 100), {"units": "m"}),
            "species": ("species", ["N2", "O2"])
        }
    )
    assert check(p) is False

    # profile with a missing dimension (species)
    p = xr.Dataset(
        data_vars={
            "pressure": ("altitude", np.random.rand(100), {"units": "Pa"}),
            "temperature": ("altitude", np.random.rand(100), {"units": "K"}),
            "number_density": ("altitude", np.random.rand(100), {"units": "m^-3"}),
            "total_number_density": ("altitude", np.random.rand(100), {"units": "m^-3"}),
            "upper_bound": ("altitude", np.random.rand(100), {"units": "m"}),
            "lower_bound": ("altitude", np.random.rand(100), {"units": "m"})},
        coords={
            "altitude": ("altitude", np.linspace(0, 100e3, 100), {"units": "m"}),
        }
    )
    assert check(p) is False

    # profile with missing coordinate (altitude)
    p = xr.Dataset(
        data_vars={
            "pressure": ("altitude", np.random.rand(100), {"units": "Pa"}),
            "temperature": ("altitude", np.random.rand(100), {"units": "K"}),
            "number_density": (("species", "altitude"), np.random.rand(2, 100), {"units": "m^-3"}),
            "total_number_density": ("altitude", np.random.rand(100), {"units": "m^-3"}),
            "upper_bound": ("altitude", np.random.rand(100), {"units": "m"}),
            "lower_bound": ("altitude", np.random.rand(100), {"units": "m"})},
        coords={
            "species": ("species", ["N2", "O2"])
        }
    )
    assert check(p) is False

    # good profile
    p = xr.Dataset(
        data_vars={
            "pressure": ("altitude", np.random.rand(100), {"units": "Pa"}),
            "temperature": ("altitude", np.random.rand(100), {"units": "K"}),
            "number_density": (("species", "altitude"), np.random.rand(2, 100), {"units": "m^-3"}),
            "total_number_density": ("altitude", np.random.rand(100), {"units": "m^-3"}),
            "upper_bound": ("altitude", np.random.rand(100), {"units": "m"}),
            "lower_bound": ("altitude", np.random.rand(100), {"units": "m"})},
        coords={
            "altitude": ("altitude", np.linspace(0, 100e3, 100), {"units": "m"}),
            "species": ("species", ["N2", "O2"])
        }
    )
    assert check(p) is True


def test_us76():
    # default constructor
    profile = us76()

    assert profile["lower_bound"].values[0] == 0.
    assert profile["upper_bound"].values[-1] == 100000.
    assert profile.dims["altitude"] == 50
    assert profile.dims["species"] == 12

    for var in ["temperature", "pressure", "number_density",
                "total_number_density", "upper_bound", "lower_bound"]:
        assert var in profile

    # custom atmosphere height
    profile = us76(height=Q_("120 kilometers"))

    assert profile.dims["altitude"] == 50
    assert profile["lower_bound"].values[0] == 0.
    assert profile["upper_bound"].values[-1] == 120000.
    assert profile.dims["species"] == 12

    for var in ["temperature", "pressure", "number_density",
                "total_number_density", "upper_bound", "lower_bound"]:
        assert var in profile

    # custom number of layers
    profile = us76(height=Q_("120 kilometers"), n_layers=36)

    assert profile.dims["altitude"] == 36
    assert profile["lower_bound"].values[0] == 0.
    assert profile["upper_bound"].values[-1] == 120000.
    assert profile.dims["species"] == 12

    for var in ["temperature", "pressure", "number_density",
                "total_number_density", "upper_bound", "lower_bound"]:
        assert var in profile

    profile = us76(height=Q_("120 kilometers"), n_layers=1)

    assert profile.dims["altitude"] == 1
    assert profile["lower_bound"].values[0] == 0.
    assert profile["upper_bound"].values[-1] == 120000.
    assert profile.dims["species"] == 12

    for var in ["temperature", "pressure", "number_density",
                "total_number_density", "upper_bound", "lower_bound"]:
        assert var in profile

    # invalid height
    with pytest.raises(ValueError):
        us76(height=-5000)

    with pytest.raises(ValueError):
        us76(height=1500000)

    # invalid number of layers
    with pytest.raises(ValueError):
        us76(n_layers=0)

    with pytest.raises(ValueError):
        us76(n_layers=-10)
