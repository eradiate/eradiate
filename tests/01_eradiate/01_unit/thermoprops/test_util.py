import numpy as np
import pytest
import xarray as xr

import eradiate.thermoprops.us76 as us76
import eradiate.thermoprops.util as util
from eradiate import unit_registry as ureg


def test_compute_column_number_density():
    # compute correctly (regular altitude mesh)
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", np.arange(0, 3) + 0.5, {"units": "m"}),
            "z_level": ("z_level", np.arange(0, 4), {"units": "m"}),
            "species": ("species", ["H2O"], {}),
        },
    )
    assert util.column_number_density(ds, "H2O") == ureg.Quantity(6, "m^-2")

    # compute correctly (irregular altitude mesh)
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O"], {}),
        },
    )
    assert util.column_number_density(ds, "H2O") == ureg.Quantity(26, "m^-2")

    # compute correctly (mutiple species)
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.5 * np.ones((2, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {}),
        },
    )
    assert util.column_number_density(ds, "H2O") == ureg.Quantity(13, "m^-2")
    assert util.column_number_density(ds, "O3") == ureg.Quantity(13, "m^-2")


def test_column_mass_density():
    # returns a quantity.
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", np.arange(0, 3) + 0.5, {"units": "m"}),
            "z_level": ("z_level", np.arange(0, 4), {"units": "m"}),
            "species": ("species", ["H2O"], {}),
        },
    )
    assert isinstance(util.column_mass_density(ds, "H2O"), ureg.Quantity)


def test_compute_number_density_at_surface():
    # compute correctly
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.6 * np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", np.arange(0, 3) + 0.5, {"units": "m"}),
            "z_level": ("z_level", np.arange(0, 4), {"units": "m"}),
            "species": ("species", ["H2O"], {}),
        },
    )
    value = util.number_density_at_surface(ds, "H2O")
    assert value == ureg.Quantity(0.6, "m^-3")


def test_mass_density_at_surface():
    # returns a quantity
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), np.ones((1, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", np.arange(0, 3) + 0.5, {"units": "m"}),
            "z_level": ("z_level", np.arange(0, 4), {"units": "m"}),
            "species": ("species", ["H2O"], {}),
        },
    )
    assert isinstance(util.mass_density_at_surface(ds, "H2O"), ureg.Quantity)


def test_scaling_factor():
    """Returns correct value."""

    assert np.isclose(
        util._scaling_factor(
            initial_amount=1.0 / ureg.m**3,
            target_amount=10.0 / ureg.m**3,
        ),
        10.0,
    )

    assert np.isclose(
        util._scaling_factor(
            initial_amount=1.0 / ureg.m**3,
            target_amount=0.0 / ureg.m**3,
        ),
        0.0,
    )


def test_scaling_factor_invalid():
    """Raises when both initial and target amounts are zero."""
    with pytest.raises(ValueError):
        util._scaling_factor(
            initial_amount=0.0 / ureg.m**3,
            target_amount=1.0 / ureg.m**3,
        )


def test_compute_scaling_factors():
    # compute correctly
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.5 * np.ones((2, 3)), {"units": ""}),
            "n": ("z_layer", np.arange(1, 4), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {}),
        },
    )

    factors = util.compute_scaling_factors(
        ds=ds, concentration={"H2O": ureg.Quantity(26.0, "m^-2")}
    )
    assert factors["H2O"] == 2.0

    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.5 * np.ones((2, 3)), {"units": ""}),
            "n": ("z_layer", 2.687e20 * np.ones(3), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {}),
        },
    )

    factors = util.compute_scaling_factors(
        ds=ds, concentration={"O3": ureg.Quantity(10, "dobson_units")}
    )
    assert np.isclose(factors["O3"], 2.0, rtol=1e-9)


def test_human_readable():
    assert util.human_readable(["a", "b", "c"]) == "a, b and c"


def test_rescale_concentration():
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.1 * np.ones((2, 3)), {"units": ""}),
            "n": ("z_layer", 2.687e20 * np.ones(3), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {}),
        },
        attrs=dict(
            history="",
        ),
    )
    initial_ozone_amount = util.column_number_density(ds=ds, species="O3")
    new_ozone_amount = ureg.Quantity(1, "dobson_units")
    factors = util.compute_scaling_factors(
        ds=ds, concentration={"O3": new_ozone_amount}
    )
    rescaled_ds = util.rescale_concentration(ds=ds, factors=factors, inplace=False)
    ds_ozone_amount = util.column_number_density(ds=ds, species="O3")
    ozone_amount = util.column_number_density(ds=rescaled_ds, species="O3")
    assert ds_ozone_amount == initial_ozone_amount
    assert ozone_amount.to("dobson_unit") == new_ozone_amount

    # too large rescaling factors raise error
    new_ozone_amount = ureg.Quantity(10, "dobson_units")
    factors = util.compute_scaling_factors(
        ds=ds, concentration={"O3": new_ozone_amount}
    )
    with pytest.raises(ValueError):
        util.rescale_concentration(ds=ds, factors=factors, inplace=False)


def test_interpolate():
    # column amounts are conserved
    ds = xr.Dataset(
        data_vars={
            "mr": (("species", "z_layer"), 0.1 * np.ones((2, 3)), {"units": ""}),
            "n": ("z_layer", 2.687e20 * np.ones(3), {"units": "m^-3"}),
        },
        coords={
            "z_layer": ("z_layer", [0.5, 2.0, 6.5], {"units": "m"}),
            "z_level": ("z_level", [0.0, 1.0, 3.0, 10.0], {"units": "m"}),
            "species": ("species", ["H2O", "O3"], {}),
        },
        attrs=dict(
            history="",
        ),
    )
    initial_amounts = {s: util.column_number_density(ds, s) for s in ds.species.values}
    interpolated = util.interpolate(
        ds=ds, method="linear", z_level=np.linspace(0, 8, 9), conserve_columns=True
    )
    amounts = {
        s: util.column_number_density(interpolated, s)
        for s in interpolated.species.values
    }
    for s in ds.species.values:
        assert np.isclose(amounts[s], initial_amounts[s], rtol=1e-9)


def test_water_vapor_saturation_pressure():
    # values are correct, reference value from:
    # https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-d_599.html
    t = ureg.Quantity(18, "celsius")
    p = util.water_vapor_saturation_pressure(t=t)
    assert np.isclose(p, ureg.Quantity(2.065, "kPa"), rtol=0.1)

    # accepts temperature below freezing point
    t = ureg.Quantity(-10, "celsius")
    p = util.water_vapor_saturation_pressure(t=t)


def test_equilibrium_water_vapor_fraction():
    # raises when equilibrium does not exist
    with pytest.raises(ValueError):
        util.equilibrium_water_vapor_fraction(
            p=ureg.Quantity(3, "kPa"), t=ureg.Quantity(50, "celsius")
        )
        util.equilibrium_water_vapor_fraction(
            p=ureg.Quantity(3, "kPa"), t=ureg.Quantity(-10, "celsius")
        )
        util.equilibrium_water_vapor_fraction(
            p=ureg.Quantity(100, "Pa"), t=ureg.Quantity(10, "celsius")
        )
        util.equilibrium_water_vapor_fraction(
            p=ureg.Quantity(100, "Pa"), t=ureg.Quantity(-10, "celsius")
        )
        util.equilibrium_water_vapor_fraction(
            p=ureg.Quantity(1, "bar"), t=ureg.Quantity(120, "celsius")
        )
    # values are in [0, 1]
    value = util.equilibrium_water_vapor_fraction(
        p=ureg.Quantity(90, "kPa"), t=ureg.Quantity(20, "celsius")
    )
    assert 0.0 <= value <= 1.0


def test_find_regular_params_gcd():
    with pytest.raises(ValueError):
        util._find_regular_params_gcd([1, 2, 3], -1.0)

    with pytest.raises(ValueError):
        util._find_regular_params_gcd([1, 2, 3], 0.0)

    assert util._find_regular_params_gcd(np.arange(10)) == (10, 1)

    mesh = [0.0, 1200.0, 2500.0, 5300.0, 10000.0, 25500.0]
    assert util._find_regular_params_gcd(mesh) == (256, 100)
    assert util._find_regular_params_gcd(mesh, unit_number=1000.0) == (26, 1000)


def test_to_regular():
    # check that the bounds are kept the same
    mesh = [0.0, 1200.0, 2500.0, 5300.0, 10000.0, 25500.0]
    regular = util._to_regular(mesh, atol=1000)
    assert regular[0] == mesh[0]
    assert regular[-1] == mesh[-1]


# TODO: test further
def test_make_profile_regular():
    irregular_profile = us76.make_profile(
        levels=ureg.Quantity([0.0, 2.0, 8.0, 10.0, 16.0, 23.0, 32.0, 47.0, 50.0], "km")
    )

    regular_profile = util.make_profile_regular(irregular_profile, atol=1000.0)
