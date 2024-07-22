from __future__ import annotations

import typing as t
from numbers import Number

import attrs
import numpy as np

from .attrs import AUTO
from .units import PhysicalQuantity
from .units import unit_registry as ureg


def is_scalar(_, attribute, value):
    """
    Validate iff value is of scalar type in the sense of Numpy.

    Raises
    ------
    TypeError
        If the value is not scalar.
    """
    if not np.isscalar(value):
        raise TypeError(f"'{attribute.name}' must be scalar, got {value}")


def is_number(_, attribute, value):
    """
    Validate iff value is of a numeric type.

    Raises
    ------
    TypeError
        If the value is not a :class:`Number`.
    """
    if not isinstance(value, Number):
        raise TypeError(
            f"{attribute.name} must be a real number, "
            f"got {value} which is a {value.__class__}"
        )


def is_vector3(instance, attribute, value):
    """
    Validate iff value can be converted to a (3,) Numpy array.

    Raises
    ------
    TypeError
        If value cannot be converted to a (3,) :class:`numpy.ndarray`.
    """
    return attrs.validators.deep_iterable(
        member_validator=is_number, iterable_validator=has_len(3)
    )(instance, attribute, value)


def is_positive(_, attribute, value):
    """
    Validate iff value is a positive number.

    Raises
    ------
    ValueError
        If the value is not positive or zero.
    """
    if value < 0.0:
        raise ValueError(f"{attribute} must be positive or zero, got {value}")


def all_positive(_, attribute, value):
    """
    Validate iff value is a vector with all its values positive.

    Raises
    ------
    ValueError
        If not all values are positive.
    """
    if isinstance(value, ureg.Quantity):
        value = value.magnitude
    if np.any(np.array(value) < 0):
        raise ValueError(f"{attribute} must be all positive or zero, got {value}")


def all_strictly_positive(_, attribute, value):
    """
    Validate iff value is a vector with all its values strictly positive.

    Raises
    ------
    ValueError
        If not all values are strictly positive.
    """
    if isinstance(value, ureg.Quantity):
        value = value.magnitude
    if np.any(np.array(value) <= 0):
        raise ValueError(f"{attribute} must be all strictly positive, got {value}")


def path_exists(_, attribute, value):
    """
    Validate iff initializer is called with a value defining a path to an
    existing location.

    Raises
    ------
    FileNotFoundError
        If the value is not a :class:`pathlib.Path` which points to an existing
        location.
    """
    if not value.exists():
        raise FileNotFoundError(
            f"{attribute} points to '{str(value)}' (path does not exist)"
        )


def is_file(_, attribute, value):
    """
    Validate iff initializer is called with a value defining a path to an
    existing file.

    Raises
    ------
    FileNotFoundError
        If the value is not a :class:`pathlib.Path` which points to an existing
        file.
    """
    if not value.is_file():
        raise FileNotFoundError(
            f"{attribute.name} points to '{str(value)}' (not a file)"
        )


def is_dir(_, attribute, value):
    """
    Validate iff initializer is called with a value defining a path to an
    existing directory.

    Raises
    ------
    FileNotFoundError
        If the value is not a :class:`pathlib.Path` which points to an existing
        directory.
    """
    if not value.is_dir():
        raise FileNotFoundError(
            f"{attribute.name} points to '{str(value)}' (not a directory)"
        )


def is_sorted(
    order: t.Literal["ascending", "descending"] = "ascending", strict: bool = False
):
    """
    Validate iff a 1D array is sorted.

    Parameters
    ----------
    order : {"ascending", "descending"}, default: "ascending"
        Order expected for the array.

    strict : bool, default: False
        If ``True``, check for strict inequality.
    """
    if order == "ascending":
        if strict:

            def cmp(a):
                return np.all(a[:-1] < a[1:])
        else:

            def cmp(a):
                return np.all(a[:-1] <= a[1:])
    else:
        if strict:

            def cmp(a):
                return np.all(a[:-1] > a[1:])
        else:

            def cmp(a):
                return np.all(a[:-1] >= a[1:])

    def f(_, attribute, value):
        if not cmp(value):
            raise ValueError(f"while validating {attribute}: array must be sorted")

    return f


def has_len(size: int):
    """
    Validate iff value is a vector with specified length.

    Parameters
    ----------
    size : int
        Expected size of the validated value.

    Raises
    ------
    ValueError
        If the value does not have the expected size.
    """

    def f(_, attribute, value):
        if len(value) != size:
            raise ValueError(
                f"{attribute} must be have length {size}, "
                f"got {value} of length {len(value)}"
            )

    return f


def has_quantity(quantity: PhysicalQuantity | str | None):
    """
    Validate iff initializer is called with a value  featuring a ``quantity``
    field set to an expected value.

    Parameters
    ----------
    quantity : :class:`.PhysicalQuantity` or str or None
        Expected quantity field.

    Raises
    ------
    ValueError
        If the value's ``quantity`` field does not match the expected value.
    """
    if quantity is not None:
        quantity = PhysicalQuantity(quantity)

    def f(_, attribute, value):
        if value.quantity != quantity:
            raise ValueError(
                f"incompatible quantity '{value.quantity}' "
                f"used to set field '{attribute.name}' "
                f"(allowed: '{quantity}')"
            )

    return f


def on_quantity(wrapped_validator: t.Callable):
    """
    Apply a validator to the magnitude of a quantity.

    Parameters
    ----------
    wrapped_validator : callable
        The validator applied to the value's magnitude.
    """

    def f(instance, attribute, value):
        if isinstance(value, ureg.Quantity):
            return wrapped_validator(instance, attribute, value.magnitude)
        else:
            return wrapped_validator(instance, attribute, value)

    return f


def auto_or(*wrapped_validators):
    """
    Allow for an attribute to be set to :class:`.AUTO`.

    Parameters
    ----------
    *wrapped_validators : callable
        Validators to be applied to values not equal to :class:`.AUTO`.
    """

    def f(instance, attribute, value):
        if value is AUTO:
            return

        for validator in wrapped_validators:
            validator(instance, attribute, value)

    return f
