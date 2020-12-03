import numpy as np
import pytest

from eradiate.scenes.atmosphere.thermophysics.us76 import make_profile
from eradiate.scenes.atmosphere.thermophysics.util import (
    make_profile_regular,
    _find_regular_params_gcd,
    _to_regular,
    profile_dataset_spec
)
from eradiate.util.units import ureg
Q_ = ureg.Quantity


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
        levels=Q_([0., 2., 8., 10., 16., 23., 32., 47., 50.], "km"))

    regular_profile = make_profile_regular(irregular_profile, atol=1000.)
    regular_profile.ert.validate_metadata(profile_dataset_spec)

# TODO: test rescale_co2
