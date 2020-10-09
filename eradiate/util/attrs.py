"""attrs-based utility classes and functions"""

import enum
from functools import lru_cache

import attr

from . import always_iterable
from .units import compatible, ureg


# ------------------------------------------------------------------------------
#                            Unit support extension
# ------------------------------------------------------------------------------

class MKey(enum.Enum):
    """Attribute metadata keys.

    These Enum values should be used as metadata attribute keys.
    """
    has_unit = enum.auto()  #: Field supports Pint units


def unit_enabled(cls):
    """This decorator marks a class as unit-enabled.

    Upon class definition (import), this function checks that every field marked
    as unit-enabled with :func:`attrib`'s ``has_unit`` parameter has a
    corresponding unit field with the same name and a ``_unit`` suffix.

    In addition, this function attaches to the decorated class a
    ``_unit_enabled_field_names()`` class method which returns the list of the
    names of fields marked as unit-enabled.
    """

    # Attach a class method which returns the list of unit-enabled attributes
    @classmethod
    @lru_cache(maxsize=None)
    def _unit_enabled_field_names(cls):
        """Returns a list of unit-enabled fields for the current class.
        Implemented using lru_cache to avoid repeatedly regenerating the list.
        """
        return [field.name
                for field in attr.fields(cls)
                if field.metadata.get(MKey.has_unit)]

    cls._unit_enabled_field_names = _unit_enabled_field_names

    # Attach an instance method which provides convenient access to unit-enabled
    # quantities
    def get_quantity(instance, attribute):
        """Construct a :class:`pint.Quantity` from a unit-enabled attribute and
        its corresponding unit.

        Parameter ``field`` (str):
            Unit-enabled attribute to query.

        Returns → :class:`pint.Quantity`:
            Constructed :class:`pint.Quantity` object.

        Raises → KeyError:
            If the requested attribute is not unit-enabled.
        """
        if attribute not in instance._unit_enabled_field_names():
            raise KeyError(f"attribute {attribute} cannot be returned as a "
                           f"pint.Quantity object")

        return ureg.Quantity(getattr(instance, attribute),
                             getattr(instance, f"{attribute}_unit"))

    setattr(cls, get_quantity.__name__, get_quantity)

    # Check if unit-enabled attributes have a corresponding unit field
    field_names = {field.name for field in attr.fields(cls)}
    for unit_enabled_field_name in cls._unit_enabled_field_names():
        unit_field_name = f"{unit_enabled_field_name}_unit"
        if unit_field_name not in field_names:
            raise AttributeError(
                f"unit-enabled field '{unit_enabled_field_name}' misses its "
                f"corresponding unit field '{unit_field_name}'"
            )

    return cls


# ------------------------------------------------------------------------------
#                             Attribute wrappers
# ------------------------------------------------------------------------------

def attrib(
        default=attr.NOTHING, validator=None, repr=True, eq=True, order=None,
        hash=None, init=True, metadata={}, type=None, converter=None,
        factory=None, kw_only=False, on_setattr=None, has_unit=False
):
    """Create a new attribute on a class.

    This wrapper extends :func:`attr.ib`: see its documentation for undocumented
    parameters.

    Parameter ``has_unit`` (bool):
        Register the created attribute as unit-enabled and expect a
        corresponding unit field.

    Returns → :class:`attr.Attribute`:
        Generated attribute field.
    """
    metadata = dict() if not metadata else metadata

    if has_unit:
        metadata[MKey.has_unit] = True

    return attr.ib(
        default=default, validator=validator, repr=repr, eq=eq, order=order,
        hash=hash, init=init, metadata=metadata, type=type, converter=converter,
        factory=factory, kw_only=kw_only, on_setattr=on_setattr
    )


def attrib_float_positive(
        default=attr.NOTHING, repr=True, eq=True, order=None, hash=None,
        init=True, metadata={}, factory=None, kw_only=False, on_setattr=None,
        has_unit=False
):
    """Define an attribute storing a positive floating-point number.

    This wrapper extends :func:`attrib`: see its documentation for undocumented
    parameters.

    Returns → :class:`attr.Attribute`:
        Generated attribute field.
    """

    metadata = dict() if not metadata else metadata

    def f(value):
        if isinstance(value, ureg.Quantity):
            return float(value.magnitude) * value.units
        else:
            return float(value)

    converter = f
    validator = validator_is_positive

    return attrib(
        default=default, validator=validator, repr=repr, eq=eq, order=order,
        hash=hash, init=init, metadata=metadata, converter=converter,
        factory=factory, kw_only=kw_only, on_setattr=on_setattr,
        has_unit=has_unit
    )


def attrib_int_positive(
        default=attr.NOTHING, repr=True, eq=True, order=None, hash=None,
        init=True, metadata={}, factory=None, kw_only=False, on_setattr=None,
        has_unit=False
):
    """Define an attribute storing a positive integer number.

    This wrapper extends :func:`attrib`: see its documentation for undocumented
    parameters.

    Returns → :class:`attr.Attribute`:
        Generated attribute field.
    """

    metadata = dict() if not metadata else metadata

    def f(value):
        if isinstance(value, ureg.Quantity):
            return int(value.magnitude) * value.units
        else:
            return int(value)

    converter = f
    validator = validator_is_positive

    return attrib(
        default=default, validator=validator, repr=repr, eq=eq, order=order,
        hash=hash, init=init, metadata=metadata, converter=converter,
        factory=factory, kw_only=kw_only, on_setattr=on_setattr,
        has_unit=has_unit
    )


def attrib_unit(
        default=attr.NOTHING, repr=True, eq=True, order=None, hash=None,
        init=True, metadata={}, kw_only=False, on_setattr=None,
        compatible_units=None
):
    """Define an attribute dedicated to storing a unit.
    This definition allows for setting a list a compatible units to specify the
    dimensionality of the expected unit.

    This wrapper extends :func:`attrib`: see its documentation for undocumented
    parameters.

    .. note::

       All units must originate from Eradiate's unit registry
       :data:`eradiate.util.units.ureg`.

    Parameter ``compatible_units`` (:class:`pint.Unit` or list[:class:`pint.Unit`] or None):
        If a unit or a list of units is passed, the generated attribute will
        validate against compatibility with those units. If ``None`` is passed,
        no unit compatibility check will be performed.

    Parameter ``attrib_kwargs``:
        All remaining keyword arguments are forwarded to :func:`attr.ib`
        function. The ``converter`` and ``validator`` are not allowed.

    Returns → :class:`attr.Attribute`:
        Generated attribute field.

    Raises → ValueError:
        If the ``default`` keyword argument is set to a unit incompatible with
        ``compatible_units``.
    """

    metadata = dict() if not metadata else metadata

    # Check that compatible units and defaults are compatible
    if compatible_units is not None:
        if default not in [None, attr.NOTHING]:
            if isinstance(default, attr.Factory):
                default_check = default.factory()
            else:
                default_check = ureg.Unit(default)

            compatible_units = tuple(always_iterable(compatible_units))
            if not any([compatible(default_check, x) for x in compatible_units]):
                raise ValueError(
                    f"incompatible allowed units "
                    f"['{''', '''.join([str(x) for x in compatible_units])}'] "
                    f"and default '{default_check}'"
                )

    # Construct attribute definition
    validators = [attr.validators.instance_of(ureg.Unit)]

    if compatible_units is not None:
        validators.append(validator_unit_compatible(compatible_units))

    converter = attr.converters.optional(ureg.Unit)
    validator = attr.validators.optional(attr.validators.and_(*validators))

    return attrib(
        default=default, validator=validator, repr=repr, eq=eq, order=order,
        hash=hash, init=init, metadata=metadata, type=type, converter=converter,
        kw_only=kw_only, on_setattr=on_setattr, has_unit=False
    )


# ------------------------------------------------------------------------------
#                                 Validators
# ------------------------------------------------------------------------------

def validator_unit_compatible(units):
    """Generates a validator which validates if value is a unit compatible with
    one of ``units``. The generated validator will raise a ``ValueError`` in
    case of failure.

    Parameter ``units`` (:class:`pint.Unit` or list[:class:`pint.Unit`]):
        Units against which validation is done.

    Returns → callable(instance, attribute, value):
        Generated validator.
    """
    units_iterable = tuple(always_iterable(units))

    def f(instance, attribute, value):
        if not any([compatible(value, x) for x in units_iterable]):
            raise ValueError(
                f"incompatible unit '{value}' used to set field '{attribute.name}' "
                f"(allowed: [{', '.join([str(x) for x in units_iterable])}])"
            )

    return f


def validator_is_number(_, attribute, value):
    """Validates if ``value`` is an ``int`` or ``float``.
    Raises a ``TypeError`` in case of failure.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{attribute} must be a real number, "
                        f"got {value} which is a {type(value)}")


def validator_is_string(_, attribute, value):
    """Validates if ``value`` is a ``str``.
    Raises a ``TypeError`` in case of failure.
    """
    if not isinstance(value, str):
        raise TypeError(f"{attribute} must be a string, "
                        f"got {value} which is a {type(value)}")


def validator_is_positive(_, attribute, value):
    """Validates if ``value`` is a positive number.
    Raises a ``ValueError`` in case of failure.
    """
    if value < 0.:
        raise ValueError(f"{attribute} must be positive or zero, "
                         f"got {value}")


def validator_has_len(size):
    """Generates a validator which validates if ``value`` is of length ``size``.
    The generated validator will raise a ``ValueError`` in
    case of failure.

    Parameter ``size`` (int):
        Size required to pass validation.

    Returns → callable(instance, attribute, value):
        Generated validator.
    """

    def f(_, attribute, value):
        if len(value) != size:
            raise ValueError(f"{attribute} must be have length {size}, "
                             f"got {value} of length {len(value)}")

    return f
