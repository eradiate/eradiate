import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import Singleton, is_vector3


def test_singleton():
    class MySingleton(metaclass=Singleton):
        pass

    my_singleton1 = MySingleton()
    my_singleton2 = MySingleton()
    assert my_singleton1 is my_singleton2


vector3_test_data = [
    ("aaa", False),
    ([0, 1], False),
    ([0, 2], False),
    ([0, 1, 2], True),
    ([0.0, 1, 2], True),
    ([0, 1, "2"], False),
    (np.array([0, 1, 2]), True),
    (np.array([0, 1]), False),
    (np.array(["0", 1, 2]), False),
    (ureg.Quantity([0, 1, 2], "m"), True),
]


@pytest.mark.parametrize("value, expected", vector3_test_data)
def test_is_vector3(value, expected):
    result = is_vector3(value)
    assert result == expected
