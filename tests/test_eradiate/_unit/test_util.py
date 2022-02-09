import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import Singleton, camel_to_snake, deduplicate, is_vector3, natsorted


def test_singleton():
    class MySingleton(metaclass=Singleton):
        pass

    my_singleton1 = MySingleton()
    my_singleton2 = MySingleton()
    assert my_singleton1 is my_singleton2


@pytest.mark.parametrize(
    "value, expected",
    [
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
    ],
)
def test_is_vector3(value, expected):
    result = is_vector3(value)
    assert result == expected


def test_natsort():
    assert natsorted(["10", "1.2", "9"]) == ["1.2", "9", "10"]
    assert natsorted(["1.2", "a1", "9"]) == ["1.2", "9", "a1"]


def test_deduplicate():
    assert deduplicate([1, 1, 2, 3], preserve_order=True) == [1, 2, 3]
    assert deduplicate([2, 1, 3, 1], preserve_order=True) == [2, 1, 3]
    assert deduplicate([2, 1, 3, 1], preserve_order=False) == [1, 2, 3]
    # Note: this latter test may not be reproducible, we might have to change it


def test_camel_to_snake():
    assert camel_to_snake("SomeKindOfThing") == "some_kind_of_thing"
