import os
from numbers import Number

import attr
import numpy as np

from .units import PhysicalQuantity
from .units import unit_registry as ureg


def is_number(_, attribute, value):
    """Validates if ``value`` is a number.
    Raises a ``TypeError`` in case of failure.
    """
    if not isinstance(value, Number):
        raise TypeError(
            f"{attribute.name} must be a real number, "
            f"got {value} which is a {value.__class__}"
        )


def is_vector3(instance, attribute, value):
    """Validates if ``value`` is convertible to a 3-vector."""
    return attr.validators.deep_iterable(
        member_validator=is_number, iterable_validator=has_len(3)
    )(instance, attribute, value)


def is_positive(_, attribute, value):
    """Validates if ``value`` is a positive number.
    Raises a ``ValueError`` in case of failure.
    """
    if value < 0.0:
        raise ValueError(f"{attribute} must be positive or zero, got {value}")


def all_positive(_, attribute, value):
    """Validates if all elements in ``value`` are positive number.
    Raises a ``ValueError`` in case of failure.
    """
    if isinstance(value, ureg.Quantity):
        value = value.magnitude
    if not np.all(np.array(value) >= 0):
        raise ValueError(f"{attribute} must be all positive or zero, got {value}")


def path_exists(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing target. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.exists():
        raise FileNotFoundError(
            f"{attribute} points to '{str(value)}' " f"(path does not exist)"
        )


def is_file(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.is_file():
        raise FileNotFoundError(
            f"{attribute} points to '{str(value)}' " f"(not a file)"
        )


def is_path(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if value is None or not os.path.exists(value):
        raise FileNotFoundError(
            f"{attribute} points to '{str(value)}' " f"(not a file)"
        )


def is_dir(_, attribute, value):
    """Validates if ``value`` is a :class:`pathlib.Path` and points to
    an existing file. Raises a ``FileNotFoundError`` otherwise.
    """
    if not value.is_dir():
        raise FileNotFoundError(
            f"{attribute} points to '{str(value)}'" f"(not a directory)"
        )


def has_len(size):
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
            raise ValueError(
                f"{attribute} must be have length {size}, "
                f"got {value} of length {len(value)}"
            )

    return f


def has_quantity(quantity):
    """Validates if the validated value has a quantity field matching the
    ``quantity`` parameter."""

    quantity = PhysicalQuantity(quantity)

    def f(_, attribute, value):
        if value.quantity != quantity:
            raise ValueError(
                f"incompatible quantity '{value.quantity}' "
                f"used to set field '{attribute.name}' "
                f"(allowed: '{quantity}')"
            )

    return f


def on_quantity(wrapped_validator):
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


def auto_or(*wrapped_validators):
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
