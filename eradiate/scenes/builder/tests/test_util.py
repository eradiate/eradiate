import pytest

from eradiate.scenes.builder.base import Float, Int, String
from eradiate.scenes.builder.util import compose, convert_to


def test_compose():
    def f1(x):
        return f"f1({x})"

    def f2(x):
        return f"f2({x})"

    # Check if composed functions are called in the expected order
    assert compose(f2, f1)("x") == "f2(f1(x))"


def test_convert_to():
    # Test for simple conversion success
    x = convert_to(Float)(1.)
    assert isinstance(x, Float)
    assert x.value == 1.

    # Test for simple conversion failure
    x = convert_to(Int)(1.)
    assert isinstance(x, float)
    assert x == 1.

    # Test with multiple possible target types
    # We expect the first successful type in the list to be returned
    x = convert_to((Int, Float))(1.)  # Conversion should succeed
    assert isinstance(x, Float)
    assert x.value == 1.
    x = convert_to((Int, Float))("a")  # Conversion should fail
    assert isinstance(x, str)
    assert x == "a"

    # Test for missing convert method detection
    with pytest.raises(TypeError):
        convert_to(float)
    with pytest.raises(TypeError):
        convert_to((float, Int))
    with pytest.raises(TypeError):
        convert_to((Int, float))
