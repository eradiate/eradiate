import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.thermoprops.us76 import make_profile
from eradiate.thermoprops.util import (
    _find_regular_params_gcd,
    _to_regular,
    equilibrium_water_vapor_fraction,
    make_profile_regular,
    profile_dataset_spec,
    water_vapor_saturation_pressure
)


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


def test_find_regular_params_gcd():
    with pytest.raises(ValueError):
        _find_regular_params_gcd([1, 2, 3], -1.)

    with pytest.raises(ValueError):
        _find_regular_params_gcd([1, 2, 3], 0.)

    assert _find_regular_params_gcd(np.arange(10)) == (10, 1)

    mesh = [0., 1200., 2500., 5300., 10000., 25500.]
    assert _find_regular_params_gcd(mesh) == (256, 100)
    assert _find_regular_params_gcd(mesh, unit_number=1000.) == (26, 1000)


def test_to_regular():
    # check that the bounds are kept the same
    mesh = [0., 1200., 2500., 5300., 10000., 25500.]
    regular = _to_regular(mesh, atol=1000)
    assert regular[0] == mesh[0]
    assert regular[-1] == mesh[-1]


# TODO: test further
def test_make_profile_regular():
    irregular_profile = make_profile(
        levels=ureg.Quantity([0., 2., 8., 10., 16., 23., 32., 47., 50.], "km"))

    regular_profile = make_profile_regular(irregular_profile, atol=1000.)
    regular_profile.ert.validate_metadata(profile_dataset_spec)

# TODO: test rescale_co2
