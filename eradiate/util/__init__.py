""" A collection of generic utilities. """

import numpy as np


def always_iterable(obj, base_type=(str, bytes)):
    """Ensure that the object it is passed is iterable.

    - If ``obj`` is iterable, return an iterator over its items.
    - If ``obj`` is not iterable, return a one-item iterable containing ``obj``.
    - If ``obj`` is `None`, return an empty iterable.

    .. note::

        Copied from the more-itertools library
        [https://github.com/more-itertools].
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
