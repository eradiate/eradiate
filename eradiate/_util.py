from numbers import Number
from typing import Sequence

import numpy
import numpy as np
import pint
import pinttr


def onedict_value(d):
    """Get the value of a single-entry dictionary."""

    if len(d) != 1:
        raise ValueError(f"dictionary has wrong length (expected 1, got {len(d)}")

    return next(iter(d.values()))


def ensure_array(x, dtype=None):
    """Ensure that passed object is a numpy array."""
    kwargs = dict(dtype=dtype) if dtype is not None else {}

    return np.array(list(pinttr.util.always_iterable(x)), **kwargs)


def is_vector3(value):
    """Returns ``True`` if a value can be interpreted as a 3-vector."""

    if isinstance(value, pint.Quantity):
        return is_vector3(value.magnitude)

    return (
        (
            isinstance(value, numpy.ndarray)
            or (isinstance(value, Sequence) and not isinstance(value, str))
        )
        and len(value) == 3
        and all(map(lambda x: isinstance(x, Number), value))
    )


class Singleton(type):
    """A simple singleton implementation.
    See also
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python.

    .. admonition:: Example

        .. code:: python

            class MySingleton(metaclass=Singleton):
                pass

            my_singleton1 = MySingleton()
            my_singleton2 = MySingleton()
            assert my_singleton1 is my_singleton2  # Should not fail
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
