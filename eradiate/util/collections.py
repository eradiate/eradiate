"""Specialised container datatypes providing alternatives to Python’s general
purpose built-in containers, dict, list, set, and tuple."""

from collections.abc import Sequence
from numbers import Number

import numpy
import pint


def is_vector3(value):
    """Returns ``True`` if a value can be interpreted as a 3-vector."""

    if isinstance(value, pint.Quantity):
        return is_vector3(value.magnitude)

    # @formatter:off
    return (
        (
            isinstance(value, numpy.ndarray) or
            (isinstance(value, Sequence) and not isinstance(value, str))
        )
        and len(value) == 3
        and all(map(lambda x: isinstance(x, Number), value))
    )
    # @formatter:on


def onedict_value(d):
    """Get the value of a single-entry dictionary."""

    if len(d) != 1:
        raise ValueError(f"dictionary has wrong length (expected 1, got {len(d)}")

    return next(iter(d.values()))
