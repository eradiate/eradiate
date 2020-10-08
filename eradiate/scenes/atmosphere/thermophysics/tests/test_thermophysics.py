import numpy as np
import pytest
import xarray as xr

from eradiate.scenes.atmosphere.thermophysics import check_vertical_profile


def test_check_vertical_profile():
    r"""We create a bunch of invalid atmospheric vertical profile data sets and
    assert that
    :func:`eradiate.scenes.atmosphere.thermophysics.is_vertical_profile` returns
    False for these data sets. We then create a valid atmospheric profile data
    set and assert that the function returns True for this data set.
    """

    # variables units and standard names
    p_meta = {"units": "Pa", "standard_name": "air_pressure"}
    t_meta = {"units": "K", "standard_name": "air_temperature"}
    n_meta = {"units": "m^-3", "standard_name": "number_density"}
    n_tot_meta = {"units": "m^-3", "standard_name": "air_number_density"}
    z_layer_meta = {"units": "m", "standard_name": "layer_altitude"}
    z_level_meta = {"units": "m", "standard_name": "level_altitude"}
    species_meta = {"units": "", "standard_name": "species"}

    data_set_metadata = {
        "convention": "CF-1.8",
        "title": "U.S. Standard Atmosphere 1976",
        "history":
            f"2020-08-13 12:00:01 - data creation - "
            f"eradiate.scenes.atmosphere.us76.create",
        "source": f"eradiate, version 0.0.1",
        "references":
            "U.S. Standard Atmosphere, 1976, NASA-TM-X-74335, NOAA-S/T-76-1562"
    }

    # profile with a missing data variable (total number density)
    p = xr.Dataset(
        data_vars={
            "p": ("z_layer", np.random.rand(100), p_meta),
            "t": ("z_layer", np.random.rand(100), t_meta),
            "n": (("species", "z_layer"), np.random.rand(2, 100), n_meta)
        },
        coords={
            "z_layer": ("z_layer", np.linspace(0.5, 99.5, 100), z_layer_meta),
            "z_level": ("level", np.linspace(0., 100., 101), z_level_meta),
            "species": ("species", ["N2", "O2"], species_meta)
        },
        attrs=data_set_metadata
    )
    with pytest.raises(ValueError):
        check_vertical_profile(p)

    # profile with a missing data coordinate (species)
    p = xr.Dataset(
        data_vars={
            "p": ("z_layer", np.random.rand(100), p_meta),
            "t": ("z_layer", np.random.rand(100), t_meta),
            "n_tot": ("z_layer", np.random.rand(100), n_tot_meta),
            "n": (("species", "z_layer"), np.random.rand(2, 100), n_meta)
        },
        coords={
            "z_layer": ("z_layer", np.linspace(0.5, 99.5, 100), z_layer_meta),
            "z_level": ("level", np.linspace(0., 100., 101), z_level_meta)
        },
        attrs=data_set_metadata
    )
    with pytest.raises(ValueError):
        check_vertical_profile(p)

    # profile with missing (mute) data coordinate (z_level)
    p = xr.Dataset(
        data_vars={
            "p": ("z_layer", np.random.rand(100), p_meta),
            "t": ("z_layer", np.random.rand(100), t_meta),
            "n_tot": ("z_layer", np.random.rand(100), n_tot_meta),
            "n": (("species", "z_layer"), np.random.rand(2, 100), n_meta)
        },
        coords={
            "z_layer": ("z_layer", np.linspace(0.5, 99.5, 100), z_layer_meta),
            "species": ("species", ["N2", "O2"], species_meta)
        },
        attrs=data_set_metadata
    )

    # good profile
    p = xr.Dataset(
        data_vars={
            "p": ("z_layer", np.random.rand(100), p_meta),
            "t": ("z_layer", np.random.rand(100), t_meta),
            "n_tot": ("z_layer", np.random.rand(100), n_tot_meta),
            "n": (("species", "z_layer"), np.random.rand(2, 100), n_meta)
        },
        coords={
            "z_layer": ("z_layer", np.linspace(0.5, 99.5, 100), z_layer_meta),
            "z_level": ("z_level", np.linspace(0., 100., 101), z_level_meta),
            "species": ("species", ["N2", "O2"], species_meta)
        },
        attrs=data_set_metadata
    )
    check_vertical_profile(p)
