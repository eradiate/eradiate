import numpy as np
import pytest

from eradiate.mesh import to_regular, find_regular_params_gcd


def test_find_regular_params_gcd():
    with pytest.raises(ValueError):
        find_regular_params_gcd([1, 2, 3], -1.)

    with pytest.raises(ValueError):
        find_regular_params_gcd([1, 2, 3], 0.)

    assert find_regular_params_gcd(np.arange(10)) == (10, 1)

    mesh = [0., 1200., 2500., 5300., 10000., 25500.]
    assert find_regular_params_gcd(mesh) == (256, 100)
    assert find_regular_params_gcd(mesh, unit_number=1000.) == (26, 1000)


def test_to_regular():
    # check that the bounds are kept the same
    mesh = [0., 1200., 2500., 5300., 10000., 25500.]
    regular = to_regular(mesh, atol=1000)
    assert regular[0] == mesh[0]
    assert regular[-1] == mesh[-1]
