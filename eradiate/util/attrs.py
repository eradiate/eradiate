"""attrs-based utility classes and functions"""

import enum
from copy import copy
from functools import lru_cache

import attr
import numpy as np
import pint

from . import always_iterable
from .exceptions import UnitsError
from .units import PhysicalQuantity, compatible as iscompatible
from .units import ensure_units, ureg


# ------------------------------------------------------------------------------
#                            Unit support extension
# ------------------------------------------------------------------------------

class MKey(enum.Enum):
    """Attribute metadata keys.

    These Enum values should be used as metadata attribute keys.
    """
    SUPPORTS_UNITS = enum.auto()  #: Field supports units (used for dict-based units specification)
    COMPATIBLE_UNITS = enum.auto()  #: Units compatible with this field


def unit_enabled(cls):
    """This decorator marks a class as unit-enabled.

    Upon class definition (import), this function attaches to the decorated class:

    * a ``_fields_supporting_units()`` class method which returns a dictionary
      mapping the names of fields marked as unit-enabled with their compatible
      units;
    * a ``from_dict()`` class method which enables instantiation from a
      dictionary with automatic handling of an associated unit field.

    This decorator must be applied to take advantage of the
    :func:`attrib_quantity` helper function.

    Returns → type:
        Updated class.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def fields_supporting_units(wrapped_cls):
        """Return a tuple with names of attributes supporting units. Implemented
        using :func:`functools.lru_cache()` to avoid repeatedly regenerating the
        list."""
        return tuple(field.name for field in attr.fields(wrapped_cls)
                     if field.metadata.get(MKey.SUPPORTS_UNITS))

    setattr(cls, "_fields_supporting_units", fields_supporting_units)

    @classmethod
    @lru_cache(maxsize=None)
    def fields_compatible_units(wrapped_cls):
        """Return a map with unit-enabled attribute field names and associated
        compatible units. Implemented using :func:`functools.lru_cache()`
        to avoid repeatedly regenerating the list."""
        return {field.name: field.metadata.get(MKey.COMPATIBLE_UNITS)
                for field in attr.fields(wrapped_cls)
                if field.metadata.get(MKey.COMPATIBLE_UNITS)}

    setattr(cls, "_fields_compatible_units", fields_compatible_units)

    @classmethod
    def from_dict(wrapped_cls, d):
        """Create from a dictionary. This class method will additionally
        pre-process the passed dictionary to merge any field with an
        associated ``"_units"`` field into a :class:`pint.Quantity` container.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → wrapped_cls:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = copy(d)

        for field in wrapped_cls._fields_supporting_units():
            # Fetch user-specified unit if any
            try:
                field_units = d_copy.pop(f"{field}_units")
            except KeyError:
                # If no unit is specified, don't attempt conversion and let the
                # constructor take care of it
                continue

            # If a unit is found, try to apply it
            # Bonus: if a unit field *and* a quantity were found, we convert the
            # quantity to the unit
            field_value = d_copy[field]
            d_copy[field] = ensure_units(field_value, field_units, convert=True)

        # Perform object creation
        return wrapped_cls(**d_copy)

    setattr(cls, "from_dict", from_dict)

    return cls


# ------------------------------------------------------------------------------
#                             Attribute wrappers
# ------------------------------------------------------------------------------


def attrib_quantity(
        default=attr.NOTHING, validator=None, repr=True, eq=True, order=None,
        hash=None, init=True, metadata={}, type=None, converter=None,
        factory=None, kw_only=False, on_setattr=attr.NOTHING,
        units_compatible=None, units_add_converter=True, units_add_validator=True
):
    """Create a new attribute on a class.

    .. warning::

       This function only works with attrs classes decorated with
       :func:`unit_enabled`.

    The created attribute is marked as
    supporting units and can therefore can applied units when created from a
    dictionary. See :meth:`.SceneElement.from_dict` for additional information.

    This wrapper extends :func:`attr.ib`: see its documentation for undocumented
    parameters.

    Parameter ``units_compatible`` (callable or :class:`pint.Unit` or str or None):
        If a :class:`pint.Unit` or a string is passed, the field will be
        attached  ``units_compatible`` as compatible units. If a callable is
        passed, the field will be  attached ``units_compatible()`` as compatible
        units. If ``None`` is passed, no compatible units will be attached to
        the field.

        In practice, compatible units are use to automatically generate
        converters and validators. See the ``units_add_converter`` and
        ``units_add_validator`` parameters.

    Parameter ``units_add_converter`` (bool):
        If ``True``, a simple converter is appended to the conversion pipeline
        define with the ``converter`` parameter. This converter ensures that the
        field is wrapped in ``units_compatible`` (using :func:`.ensure_units`).
        If ``units_compatible`` is a callable (*e.g.* created using
        :meth:`.DefaultUnits.generator`), it will be evaluated by
        :func:`.ensure_units`. If ``default`` is ``None``, this converter will
        also be made :func:`~attr.converters.optional`.

        .. warning::

           The unit converter is appended **at the end** of the conversion
           pipeline: unit conversion is the last conversion operation performed.
           If this is not suitable, this argument should be set to ``False`` and
           a customised converter should be used as the ``converter`` parameter.

    Parameter ``units_add_validator`` (bool):
        If ``True``,  :func:`validator_has_compatible_units` is appended to the
        list of validators. If ``default`` is ``None``, this validator will
        also be made :func:`~attr.validators.optional`.

    Parameter ``on_setattr`` (callable or None):
        If unset and ``units_compatible`` is not ``None``, ``on_setattr`` is
        automatically set to
        ``attr.setters.pipe(attr.setters.convert, attr.setters.validate)``.
        Otherwise, it is redirected to :func:`attr.ib` without change.

    Returns → :class:`attr._make._CountingAttr`:
        Generated attribute field.
    """

    # Initialise attr.ib arguments
    metadata = dict() if not metadata else metadata

    converters = []
    if converter is not None:
        converters.extend(always_iterable(converter))

    validators = []
    if validator is not None:
        validators.extend(always_iterable(validator))

    # Mark the field as having units
    metadata[MKey.SUPPORTS_UNITS] = True

    # Process declared compatible units
    if units_compatible is not None:
        # Set field metadata
        if callable(units_compatible):
            metadata[MKey.COMPATIBLE_UNITS] = ureg.Unit(units_compatible())
        else:
            metadata[MKey.COMPATIBLE_UNITS] = ureg.Unit(units_compatible)

        # Set field converter
        if units_add_converter:
            if default is None:
                converters.append(attr.converters.optional(
                    lambda x: ensure_units(x, units_compatible)
                ))
            else:
                converters.append(lambda x: ensure_units(x, units_compatible))

        # Set field validator
        if units_add_validator:
            if default is None:
                validators.append(attr.validators.optional(
                    validator_has_compatible_units
                ))
            else:
                validators.append(validator_has_compatible_units)

        # Ensure that unit conversion and validation is carried out upon setting
        if on_setattr is attr.NOTHING:
            on_setattr = attr.setters.pipe(attr.setters.convert, attr.setters.validate)

    # If on_setattr hasn't been set because units_compatible is unset, we set it
    # to a valid value
    if on_setattr is attr.NOTHING:
        on_setattr = None

    return attr.ib(
        default=default, validator=validators if validators else validator,
        repr=repr, eq=eq, order=order, hash=hash, init=init, metadata=metadata,
        type=type,
        converter=converters if converters else converter,
        factory=factory, kw_only=kw_only, on_setattr=on_setattr
    )


# ------------------------------------------------------------------------------
#                                 Converters
# ------------------------------------------------------------------------------

def converter_quantity(wrapped_converter):
    """Applies a converter to the magnitude of a :class:`pint.Quantity`."""

    def f(value):
        if isinstance(value, pint.Quantity):
            return wrapped_converter(value.magnitude) * value.units
        else:
            return wrapped_converter(value)

    return f


def converter_to_units(units):
    """Returns a converter which ensures that its passed value has units
    ``units``.

    .. seealso::

       :func:`ensure_units`
    """
    return lambda x: ensure_units(x, units)


def converter_or_auto(wrapped_converter):
    """Returns a converter which executes the wrapped converter if the converted
    value is not equal to ``"auto"``; otherwise returns ``"auto"``.
    """

    def f(value):
        if value == "auto":
            return value

        return wrapped_converter(value)

    return f


# ------------------------------------------------------------------------------
#                                 Validators
# ------------------------------------------------------------------------------

def validator_has_compatible_units(instance, attribute, value):
    """Validates if ``value`` has units compatible with ``attribute``. Only
    works with unit-enabled fields created with :func:`attrib_quantity`."""
    compatible_units = instance._fields_compatible_units()[attribute.name]

    try:
        if not iscompatible(value.units, compatible_units):
            raise UnitsError(f"incompatible unit '{value.units}' "
                             f"used to set field '{attribute.name}' "
                             f"(allowed: '{compatible_units}')")
    except AttributeError:  # value.units doesn't exist
        raise UnitsError(f"unitless value '{value}' "
                         f"used to set field '{attribute.name}' "
                         f"(requires units '{compatible_units}')")


def validator_units_compatible(units):
    """Generates a validator which validates if value is a unit compatible with
    one of ``units``. The generated validator will raise a :class:`.UnitsError`
    in case of failure.

    Parameter ``units`` (:class:`pint.Unit` or list[:class:`pint.Unit`]):
        Units against which validation is done.

    Returns → callable(instance, attribute, value):
        Generated validator.
    """
    units_iterable = tuple(always_iterable(units))

    def f(instance, attribute, value):
        if not any([iscompatible(value, x) for x in units_iterable]):
            raise UnitsError(
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
        raise ValueError(f"{attribute} must be positive or zero, got {value}")


def validator_all_positive(_, attribute, value):
    """Validates if all elements in ``value`` are positive number.
    Raises a ``ValueError`` in case of failure.
    """
    if isinstance(value, ureg.Quantity):
        value = value.magnitude
    if not np.all(np.array(value) >= 0):
        raise ValueError(f"{attribute} must be all positive or zero, got {value}")


def validator_path_exists(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing target. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.exists():
        raise FileNotFoundError(f"{attribute} points to '{str(value)}' "
                                f"(path does not exist)")


def validator_is_file(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.is_file():
        raise FileNotFoundError(f"{attribute} points to '{str(value)}' "
                                f"(not a file)")


def validator_is_dir(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.is_dir():
        raise FileNotFoundError(f"{attribute} points to '{str(value)}'"
                                f"(not a directory)")


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


def validator_has_quantity(quantity):
    """Validates if the validated value has a quantity field matching the
    ``quantity`` parameter."""

    quantity = PhysicalQuantity.from_any(quantity)

    def f(_, attribute, value):
        if value.quantity != quantity:
            raise ValueError(f"incompatible quantity '{value.quantity}' "
                             f"used to set field '{attribute.name}' "
                             f"(allowed: '{quantity}')")

    return f


def validator_quantity(wrapped_validator):
    """Applies a validator to either a value or its magnitude if it is a
    :class:`pint.Quantity` object.

    Parameter ``wrapped_validator`` (callable(instance, attribute, value)):
        A validator to wrap.

    Returns → callable(instance, attribute, value):
        Wrapped validator.
    """

    def f(instance, attribute, value):
        if isinstance(value, ureg.Quantity):
            return wrapped_validator(instance, attribute, value.magnitude)
        else:
            return wrapped_validator(instance, attribute, value)

    return f


def validator_or_auto(*wrapped_validators):
    """Validates if the validated value is ``"auto"`` or if all wrapped
    validators validate.

    .. note::
       ``wrapped_validators`` is variadic and can therefore be an arbitrary
       number of validators.
    """

    def f(instance, attribute, value):
        if value == "auto":
            return

        for validator in wrapped_validators:
            validator(instance, attribute, value)

    return f
