from eradiate.units import symbol
from eradiate.units import unit_registry as ureg


def test_symbol():
    # Convert strings
    assert symbol("metre") == "m"
    assert symbol("dimensionless") == ""
    assert symbol("") == ""

    # Convert units
    assert symbol(ureg.dimensionless) == ""
    assert symbol(ureg.m) == "m"
