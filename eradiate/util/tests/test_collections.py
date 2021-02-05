import numpy as np
import pytest

from eradiate.util.collections import is_vector3
from eradiate.util.units import ureg

vector3_test_data = [
    ("aaa", False),
    ([0, 1], False),
    ([0, 2], False),
    ([0, 1, 2], True),
    ([0., 1, 2], True),
    ([0, 1, "2"], False),
    (np.array([0, 1, 2]), True),
    (np.array([0, 1]), False),
    (np.array(["0", 1, 2]), False),
    (ureg.Quantity([0, 1, 2], "m"), True)
]


@pytest.mark.parametrize("value, expected", vector3_test_data)
def test_is_vector3(value, expected):
    result = is_vector3(value)
    assert result == expected
