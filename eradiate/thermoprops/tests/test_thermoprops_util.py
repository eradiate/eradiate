import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.thermoprops.us76 import make_profile
from eradiate.thermoprops.util import (
    compute_column_number_density,
    compute_number_density_at_surface,
    compute_scaling_factors,
    equilibrium_water_vapor_fraction,
    human_readable,
    interpolate,
    make_profile_regular,
    profile_dataset_spec,
    rescale_concentration,
    water_vapor_saturation_pressure,
)


def test_compute_column_number_density():
    # compute correctly (regular altitude mesh)
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", np.arange(0, 3) + 0.5, {"units": "m"}),
            "z_level": ("z_level", np.arange(0, 4), {"units": "m"}),
            "species": ("species", ["H2O"], {})
        }
    )
    assert compute_column_number_density(ds, "H2O") == ureg.Quantity(6, "m^-2")

    # compute correctly (irregular altitude mesh)
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O"], {})
        }
    )
    assert compute_column_number_density(ds, "H2O") == ureg.Quantity(26, "m^-2")

    # compute correctly (mutiple species)
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.5 * np.ones((2,3)),
                {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {})
        }
    )
    assert compute_column_number_density(ds, "H2O") == ureg.Quantity(13, "m^-2")
    assert compute_column_number_density(ds, "O3") == ureg.Quantity(13, "m^-2")


def test_compute_number_density_at_surface():
    # compute correctly
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.6 * np.ones((1, 3)),
                {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", np.arange(0, 3) + 0.5, {"units": "m"}),
            "z_level": ("z_level", np.arange(0, 4), {"units": "m"}),
            "species": ("species", ["H2O"], {})
        }
    )
    value = compute_number_density_at_surface(ds, "H2O")
    assert value == ureg.Quantity(0.6, "m^-3")


def test_compute_scaling_factors():
    # compute correctly
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.5 * np.ones((2,3)),
                {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {})
        }
    )
    
    factors = compute_scaling_factors(
        ds=ds,
        concentration={
            "H2O": ureg.Quantity(26.0, "m^-2")
        }
    )
    assert factors["H2O"] == 2.0

    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.5 * np.ones((2,3)),
                {"units": ""}),
            "n": ("z_layer", 2.687e20 * np.ones(3), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {})
        }
    )

    factors = compute_scaling_factors(
        ds=ds,
        concentration = {
            "O3": ureg.Quantity(10, "dobson_units")
        }
    )
    assert np.isclose(factors["O3"], 2.0, rtol=1e-9)


def test_human_readable():
    assert human_readable(["a", "b", "c"]) == "a, b and c"


def test_rescale_concentration():
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.1 * np.ones((2, 3)),
                {"units": ""}),
            "n": ("z_layer", 2.687e20 * np.ones(3), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {})
        },
        attrs=dict(
            history="",
        )
    )
    initial_ozone_amount = compute_column_number_density(
        ds=ds,
        species="O3"
    )
    new_ozone_amount = ureg.Quantity(1, "dobson_units")
    factors = compute_scaling_factors(
        ds=ds,
        concentration = {
            "O3": new_ozone_amount
        }
    )
    rescaled_ds = rescale_concentration(
        ds=ds,
        factors=factors,
        inplace=False
    )
    ds_ozone_amount = compute_column_number_density(
        ds=ds,
        species="O3"
    )
    ozone_amount = compute_column_number_density(
        ds=rescaled_ds,
        species="O3"
    )
    assert ds_ozone_amount == initial_ozone_amount
    assert ozone_amount.to("dobson_unit") == new_ozone_amount

    # too large rescaling factors raise error
    with pytest.raises(ValueError):
        new_ozone_amount = ureg.Quantity(10, "dobson_units")
        factors = compute_scaling_factors(
            ds=ds,
            concentration = {
                "O3": new_ozone_amount
            }
        )
        rescaled_ds = rescale_concentration(
            ds=ds,
            factors=factors,
            inplace=False
        ) 


def test_interpolate():
    # column amounts are conserved
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.1 * np.ones((2, 3)),
                {"units": ""}),
            "n": ("z_layer", 2.687e20 * np.ones(3), {"units": "m^-3"})
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {})
        },
        attrs=dict(
            history="",
        )
    )
    initial_amounts = {
        s: compute_column_number_density(ds, s) for s in ds.species.values}
    interpolated = interpolate(
        ds=ds,
        method="linear",
        z_level=np.linspace(0, 8, 9),
        conserve_columns=True
    )
    amounts = {
        s: compute_column_number_density(interpolated, s)
        for s in interpolated.species.values
    }
    for s in ds.species.values:
        assert np.isclose(amounts[s], initial_amounts[s], rtol=1e-9)


def test_water_vapor_saturation_pressure():
    # values are correct, reference value from:
    # https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-d_599.html
    t = ureg.Quantity(18, "celsius")
    p = water_vapor_saturation_pressure(t=t)
    assert np.isclose(p, ureg.Quantity(2.065, "kPa"), rtol=0.1)

    # accepts temperature below freezing point
    t = ureg.Quantity(-10, "celsius")
    p = water_vapor_saturation_pressure(t=t)


def test_equilibrium_water_vapor_fraction():
    # raises when equilibrium does not exist
    with pytest.raises(ValueError):
        equilibrium_water_vapor_fraction(p=ureg.Quantity(3, "kPa"),
                                         t=ureg.Quantity(50, "celsius"))
        equilibrium_water_vapor_fraction(p=ureg.Quantity(3, "kPa"),
                                         t=ureg.Quantity(-10, "celsius"))
        equilibrium_water_vapor_fraction(p=ureg.Quantity(100, "Pa"),
                                         t=ureg.Quantity(10, "celsius"))
        equilibrium_water_vapor_fraction(p=ureg.Quantity(100, "Pa"),
                                         t=ureg.Quantity(-10, "celsius"))
        equilibrium_water_vapor_fraction(p=ureg.Quantity(1, "bar"),
                                         t=ureg.Quantity(120, "celsius"))
    # values are in [0, 1]
    value = equilibrium_water_vapor_fraction(p=ureg.Quantity(90, "kPa"),
                                             t=ureg.Quantity(20, "celsius"))
    assert 0. <= value <= 1.


# TODO: test further
def test_make_profile_regular():
    irregular_profile = make_profile(
        levels=ureg.Quantity([0., 2., 8., 10., 16., 23., 32., 47., 50.], "km"))

    regular_profile = make_profile_regular(irregular_profile, atol=1000.)
    regular_profile.ert.validate_metadata(profile_dataset_spec)
