import attr
import pytest

from eradiate.util.attrs import attrib, attrib_float_positive, attrib_units, unit_enabled
from eradiate.util.exceptions import UnitsError
from eradiate.util.units import config_default_units as cdu, ureg


def test_unit_enabled():
    # We expect that a missing unit attribute will raise
    with pytest.raises(AttributeError):
        @unit_enabled
        @attr.s
        class MyClassA:
            a = attrib(has_units=True)

    @unit_enabled
    @attr.s
    class MyClass:
        a = attrib(has_units=True)
        a_units = attrib_units(compatible_units=ureg.m)
        b = attrib(default=1.)

        def __attrs_post_init__(self):
            self._strip_units()

    # Check that it works when specifying separately magnitude and units
    a = MyClass(a=1., a_units=ureg.km)
    assert isinstance(a.a, float)
    assert a.get_quantity("a") == ureg.Quantity(1., ureg.km)

    with pytest.raises(AttributeError):
        a.get_quantity("b")

    with pytest.raises(AttributeError):
        a.get_quantity("c")

    # Check that conversion is correct
    a = MyClass(a=ureg.Quantity(1., ureg.km), a_units=ureg.m)
    assert a.get_quantity("a") == ureg.Quantity(1000., ureg.m)

    # Check that incompatible units will raise
    with pytest.raises(UnitsError):
        a = MyClass(a=1., a_units=ureg.s)  # Incompatible unit field parameter

    with pytest.raises(UnitsError):
        a = MyClass(a=ureg.Quantity(1., ureg.s), a_units=ureg.m)

    with pytest.raises(UnitsError):
        a = MyClass(a=ureg.Quantity(1., ureg.m / ureg.deg), a_units=ureg.m)


def test_attrib_positive_float():
    @attr.s
    class MyClass:
        a = attrib_float_positive()
        b = attrib_float_positive(default=1., has_units=True)
        b_units = attrib_units(compatible_units=ureg.m, default=ureg.m)

    o = MyClass(a=1., b=ureg.Quantity(1, ureg.m))  # Check that object constructs correctly
    assert isinstance(o.b.magnitude, float)  # Check that Pint quantities are correctly converted

    with pytest.raises(ValueError):
        MyClass(a=-1.)


def test_attrib_units():
    # Basic construct
    assert attrib_units(compatible_units=ureg.m) is not None
    # With a compatible default unit specified as a string
    assert attrib_units(compatible_units=ureg.m, default="km") is not None
    # With several units
    assert attrib_units(compatible_units=[ureg.m, ureg.s], default=ureg.km) is not None
    # With several compatible units
    assert attrib_units(compatible_units=[ureg.m, ureg.mile], default=ureg.km) is not None
    # Check that inconsistent defaults won't be allowed
    with pytest.raises(UnitsError):
        attrib_units(compatible_units=ureg.m, default=ureg.s)

    # A well-formed class
    @attr.s
    class MyClass:
        # The simplest definition: no restriction to allowed units
        # (just make sure that the field is a unit)
        unit1 = attrib_units(default=None)
        # We now allow a single unit
        unit2 = attrib_units(compatible_units=ureg.m, default=ureg.m)
        # With angle units
        unit3 = attrib_units(compatible_units=ureg.rad, default=ureg.deg)
        # With multiple compatible units
        unit4 = attrib_units(compatible_units=[ureg.m, ureg.s], default=ureg.km)
        # With a single unit and None as default
        unit5 = attrib_units(compatible_units=ureg.m, default=None)

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
    with pytest.raises(UnitsError):
        MyClass(unit2=ureg.s)
    assert MyClass(unit3=ureg.deg).unit3 == ureg.deg
    assert MyClass(unit3=ureg.rad).unit3 == ureg.rad
    assert MyClass(unit4=ureg.s).unit4 == ureg.s
    with pytest.raises(UnitsError):
        MyClass(unit4=ureg.deg)


def test_default_units():
    # Test dynamic default units
    @attr.s
    class MyClass:
        unit = attrib_units(
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
    with pytest.raises(UnitsError):
        attrib_units(
            compatible_units=ureg.s,
            default=attr.Factory(lambda: cdu.get("length"))
        )
