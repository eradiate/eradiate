import re
import typing as t
from numbers import Number

import numpy as np
import pint
import pinttr


def onedict_value(d: t.Mapping):
    """
    Get the value of a single-entry dictionary.

    Parameters
    ----------
    d : mapping
        A single-entry mapping.

    Returns
    -------
    object
        Unwrapped value.

    Raises
    ------
    ValueError
        If ``d`` has more than a single element.

    Notes
    -----
    This function is basically ``next(iter(d.values()))`` with a safeguard.

    Examples
    --------
    >>> onedict_value({"foo": "bar"})
    "bar"
    """

    if len(d) != 1:
        raise ValueError(f"dictionary has wrong length (expected 1, got {len(d)}")

    return next(iter(d.values()))


def ensure_array(value: t.Any, dtype: t.Optional[t.Any] = None) -> np.ndarray:
    """
    Convert or wrap a value in a Numpy array.

    Parameters
    ----------
    value
        Value to convert to a Numpy array

    dtype : data-type, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        A new array with the passed value.
    """
    kwargs = dict(dtype=dtype) if dtype is not None else {}

    return np.array(list(pinttr.util.always_iterable(value)), **kwargs)


def is_vector3(value: t.Any):
    """
    Check if value can be interpreted as a 3-vector.

    Parameters
    ----------
    value
        Value to be checked.

    Returns
    -------
    bool
        ``True`` if a value can be interpreted as a 3-vector.
    """

    if isinstance(value, pint.Quantity):
        return is_vector3(value.magnitude)

    return (
        (
            isinstance(value, np.ndarray)
            or (isinstance(value, t.Sequence) and not isinstance(value, str))
        )
        and len(value) == 3
        and all(map(lambda x: isinstance(x, Number), value))
    )


def natsort_alphanum_key(x):
    """
    Simple sort key natural order for string sorting. See [1]_ for details.

    See Also
    --------
    `Sorting HOWTO <https://docs.python.org/3/howto/sorting.html>`_

    References
    ----------
    .. [1] Natural sorting
       (`post on Stack Overflow <https://stackoverflow.com/a/11150413/3645374>`_).
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return tuple(convert(c) for c in re.split("([0-9]+)", x))


def natsorted(l):
    """
    Sort a list of strings with natural ordering.

    Parameters
    ----------
    l : iterable
        List to sort.

    Returns
    -------
    list
        List sorted using :func:`natsort_alphanum_key`.
    """
    return sorted(l, key=natsort_alphanum_key)


class Singleton(type):
    """
    A simple singleton implementation. See [1]_ for details.

    References
    -------
    .. [1] Creating a singleton in Python
           (`post on Stack Overflow <https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python>`_).

    Examples
    --------
    >>> class MySingleton(metaclass=Singleton):
    ... pass

    >>> my_singleton1 = MySingleton()
    >>> my_singleton2 = MySingleton()
    >>> assert my_singleton1 is my_singleton2  # Should not fail
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def str_summary_numpy(x):
    with np.printoptions(
        threshold=4, edgeitems=2, formatter={"float_kind": lambda x: f"{x:g}"}
    ):
        shape_str = ",".join(map(str, x.shape))
        prefix = f"array<{shape_str}>("
        array_str = f"{x}"

        # Indent repr if it is multiline
        split = array_str.split("\n")
        if len(split) > 1:
            array_str = ("\n" + " " * len(prefix)).join(split)

        return f"{prefix}{array_str})"
