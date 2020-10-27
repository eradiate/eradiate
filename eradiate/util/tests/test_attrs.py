import attr
import pytest

from eradiate.util.attrs import (
    attrib_quantity, unit_enabled, validator_has_len, validator_quantity
)
from eradiate.util.exceptions import UnitsError
from eradiate.util.units import config_default_units as cdu, ureg


def test_unit_support():
    # Check the unit_enabled() decorator and the attrib_quantity() attribute
    # definition wrapper

    # Check if units are correctly registered as metadata
    @unit_enabled
    @attr.s
    class MyClass:
        field_distance = attrib_quantity(
            default=ureg.Quantity(100, "km"),
            units_compatible=ureg.m
        )

        field_angle = attrib_quantity(
            default=ureg.Quantity(0, "deg"),
            units_compatible=ureg.deg
        )

        field_no_quantity = attrib_quantity(
            default=ureg.Quantity(0, "m"),
        )

    # -- This also tests if _fields_with_units() returns the expected value
    assert MyClass._fields_with_units() == {
        "field_distance": ureg.m,
        "field_angle": ureg.deg
    }

    # Check if converters are correctly set
    @unit_enabled
    @attr.s
    class MyClass:
        field_no_quantity = attrib_quantity(
            default=ureg.Quantity(100, "km"),
        )

        field_no_converter = attrib_quantity(
            default=ureg.Quantity(100, "km"),
            units_compatible=ureg.m,
            units_add_converter=False,
            units_add_validator=False,
        )

        field_static_converter = attrib_quantity(
            default=ureg.Quantity(100, "km"),
            units_compatible=ureg.m,
            units_add_converter=True,
            units_add_validator=False,
        )

        field_dynamic_converter = attrib_quantity(
            default=ureg.Quantity(100, "km"),
            units_compatible=cdu.generator("length"),
            units_add_converter=True,
            units_add_validator=False,
        )

    # -- Whatever the conversion policy, quantities used to init the object should
    #    be left intact
    with cdu.override({"length": ureg.m}):
        o = MyClass()
    assert o.field_no_quantity == ureg.Quantity(100, "km")
    assert o.field_no_quantity.units == ureg.km
    assert o.field_no_converter == ureg.Quantity(100, "km")
    assert o.field_no_converter.units == ureg.km
    assert o.field_static_converter == ureg.Quantity(100, "km")
    assert o.field_static_converter.units == ureg.km
    assert o.field_dynamic_converter == ureg.Quantity(100, "km")
    assert o.field_dynamic_converter.units == ureg.km
    # -- Converters should add units to non-quantity values
    with cdu.override({"length": ureg.km}):
        o = MyClass(
            field_no_quantity=100,
            field_no_converter=100,
            field_static_converter=100,
            field_dynamic_converter=100
        )
    assert o.field_no_quantity == 100
    assert o.field_no_converter == 100
    assert o.field_static_converter == ureg.Quantity(100, "m")
    assert o.field_dynamic_converter == ureg.Quantity(100, "km")

    # -- Check if validators are correctly set
    @unit_enabled
    @attr.s
    class MyClass:
        field_no_quantity = attrib_quantity(
            default=ureg.Quantity(100, "km"),
        )

        field_no_validator = attrib_quantity(
            default=ureg.Quantity(100, "km"),
            units_compatible=ureg.m,
            units_add_converter=True,
            units_add_validator=False,
        )

        field_unit_validator = attrib_quantity(
            default=ureg.Quantity(100, "km"),
            units_compatible=ureg.m,
            units_add_converter=True,
            units_add_validator=True,
        )

        field_multi_validator = attrib_quantity(
            default=ureg.Quantity([1, 1, 1], "km"),
            validator=validator_has_len(3),
            units_compatible=ureg.m,
            units_add_converter=True,
            units_add_validator=True,
        )

    o = MyClass()
    # -- We assume that on_setattr is set to execute converters and
    #    validators (this also tests that part of attrib_quantity())
    o.field_no_quantity = 100
    assert o.field_no_quantity == 100

    o.field_no_validator = ureg.Quantity(100, "s")  # Should not fail
    with pytest.raises(UnitsError):  # Incompatible dimensions
        o.field_unit_validator = ureg.Quantity(100, "s")
    with pytest.raises(UnitsError):  # Compatible dimensions, incompatible units
        o.field_unit_validator = ureg.Quantity(100, "m/deg")

    o.field_multi_validator = ureg.Quantity([1, 1, 1], "m")  # Should not fail
    with pytest.raises(UnitsError):  # Wrong unit
        o.field_multi_validator = ureg.Quantity([1, 1, 1], "m/deg")
    with pytest.raises(ValueError):  # Wrong size
        o.field_multi_validator = ureg.Quantity([0, 0], "m")
    with pytest.raises(ValueError):  # Wrong and unit (unit is validated last)
        o.field_multi_validator = ureg.Quantity([0, 0], "s")


def test_unit_enabled():
    # Correct field definition: this will succeed
    @unit_enabled
    @attr.s
    class MyClass:
        a = attrib_quantity(units_compatible=ureg.km)
        b = attrib_quantity(units_compatible=ureg.s)
        c = attrib_quantity()

    # Check that field metadata is correct
    assert MyClass._fields_with_units() == {
        "a": ureg.km,
        "b": ureg.s
    }

    # Check that it works when specifying separately magnitude and units
    o = MyClass(a=1., b=1., c=1.)
    assert o.a == ureg.Quantity(1., ureg.km)
    assert o.b == ureg.Quantity(1., ureg.s)
    assert o.c == 1.

    # Check that dict constructor works as expected
    @unit_enabled
    @attr.s
    class MyClass:
        a = attrib_quantity(
            default=ureg.Quantity(1, "km"),
            units_compatible=ureg.m,
            units_add_converter=True,
            units_add_validator=True
        )

    # Dict initialiser tests
    # -- Check if scalar + units yields proper field value
    o = MyClass.from_dict({"a": 1., "a_units": "m"})
    assert o.a == ureg.Quantity(1, "m")
    # -- Check if scalar is attached default units as expected
    o = MyClass.from_dict({"a": 1.})
    assert o.a == ureg.Quantity(1, "m")
    # -- Check if quantity is attached default units as expected
    o = MyClass.from_dict({"a": ureg.Quantity(10, "km")})
    assert o.a == ureg.Quantity(10, "km")
    # -- Check if the unit field can be used to force conversion of quantity
    o = MyClass.from_dict({"a": ureg.Quantity(10, "km"), "a_units": "m"})
    assert o.a == ureg.Quantity(10e3, "m")
    assert o.a.units == ureg.m


def test_validator_quantity():
    v = validator_quantity(attr.validators.instance_of(float))

    # This should succeed
    v(None, None, 1.)
    v(None, None, ureg.Quantity(1., "km"))

    # This should fail
    @attr.s
    class Attribute:  # Tiny class to pass an appropriate attribute argument
        name = attr.ib()

    attribute = Attribute(name="attribute")

    with pytest.raises(TypeError): v(None, attribute, "1.")
    with pytest.raises(TypeError): v(None, attribute, ureg.Quantity("1.", "km"))
