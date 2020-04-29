""" A collection of generic, high-level utilities. """

import numpy as np

import pint

# Pint definitions
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def always_iterable(obj, base_type=(str, bytes)):
    """
    If obj is iterable, return an iterator over its items.
    If obj is not iterable, return a one-item iterable containing obj.
    If obj is None, return an empty iterable.

    Copied from the more-itertools library [https://github.com/more-itertools]
    """
    if obj is None:
        return iter(())

    if (base_type is not None) and isinstance(obj, base_type):
        return iter((obj,))

    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))


def ensure_array(x):
    """Ensure that passed object is a numpy array."""
    return np.array(list(always_iterable(x)))
