import attr
import pytest

from eradiate.util.attrs import attrib_float_positive, attrib_unit
from eradiate.util.units import config_default_units as cdu, ureg


def test_attrib_positive_float():
    @attr.s
    class MyClass:
        a = attrib_float_positive()
        b = attrib_float_positive(default=1., has_unit=True)
        b_unit = attrib_unit(compatible_units=ureg.m, default=ureg.m)

    o = MyClass(a=1., b=ureg.Quantity(1, ureg.m))  # Check that object constructs correctly
    assert isinstance(o.b.magnitude, float)  # Check that Pint quantities are correctly converted

    with pytest.raises(ValueError):
        MyClass(a=-1.)


def test_attrib_unit():
    # Basic construct
    assert attrib_unit(compatible_units=ureg.m) is not None
    # With a compatible default unit specified as a string
    assert attrib_unit(compatible_units=ureg.m, default="km") is not None
    # With several units
    assert attrib_unit(compatible_units=[ureg.m, ureg.s], default=ureg.km) is not None
    # With several compatible units
    assert attrib_unit(compatible_units=[ureg.m, ureg.mile], default=ureg.km) is not None
    # Check that inconsistent defaults won't be allowed
    with pytest.raises(ValueError):
        attrib_unit(compatible_units=ureg.m, default=ureg.s)

    # A well-formed class
    @attr.s
    class MyClass:
        # The simplest definition: no restriction to allowed units
        # (just make sure that the field is a unit)
        unit1 = attrib_unit(default=None)
        # We now allow a single unit
        unit2 = attrib_unit(compatible_units=ureg.m, default=ureg.m)
        # With angle units
        unit3 = attrib_unit(compatible_units=ureg.rad, default=ureg.deg)
        # With multiple compatible units
        unit4 = attrib_unit(compatible_units=[ureg.m, ureg.s], default=ureg.km)
        # With a single unit and None as default
        unit5 = attrib_unit(compatible_units=ureg.m, default=None)

    # Test that the default constructor works as expected
    o = MyClass()
    assert o.unit1 is None
    assert o.unit2 == ureg.m
    assert o.unit3 == ureg.deg
    assert o.unit4 == ureg.km
    assert o.unit5 is None

    # Test that defined fields convert to units
    assert MyClass(unit1=ureg.m).unit1 == ureg.m
    assert MyClass(unit1="m").unit1 == ureg.m

    # Test that only allowed units pass (if any)
    assert MyClass(unit2=ureg.km).unit2 == ureg.km
    with pytest.raises(ValueError):
        MyClass(unit2=ureg.s)
    assert MyClass(unit3=ureg.deg).unit3 == ureg.deg
    assert MyClass(unit3=ureg.rad).unit3 == ureg.rad
    assert MyClass(unit4=ureg.s).unit4 == ureg.s
    with pytest.raises(ValueError):
        MyClass(unit4=ureg.deg)


def test_default_units():
    # Test dynamic default units
    @attr.s
    class MyClass:
        unit = attrib_unit(
            compatible_units=ureg.m,
            default=attr.Factory(lambda: cdu.get("length"))  # Defined as a lambda
        )

    o = MyClass()
    assert o.unit == ureg.m

    with cdu.override({"length": "km"}):
        assert o.unit == ureg.m  # Shouldn't change with override
        assert MyClass().unit == ureg.km  # New object's default changes with override

    # Check that default and compatible units consistency is still checked
    # properly if default is a callable
    with pytest.raises(ValueError):
        attrib_unit(
            compatible_units=ureg.s,
            default=attr.Factory(lambda: cdu.get("length"))
        )
