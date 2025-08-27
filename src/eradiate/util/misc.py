"""
A collection of tools which don't really fit anywhere else.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import typing as t
from collections import OrderedDict
from numbers import Number
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from eradiate.typing import PathLike


class cache_by_id:
    """
    Cache the result of a function based on the ID of its arguments.

    This decorator caches the value returned by the function it wraps in order
    to avoid unnecessary execution upon repeated calls with the same arguments.

    Warnings
    --------
    The main difference with
    :func:`functools.lru_cache(maxsize=1) <functools.lru_cache>` is that the
    cache is referenced by positional argument IDs instead of hashes.
    Therefore, this decorator can be used with NumPy arrays; but it's also
    unsafe, because mutating an argument won't trigger a recompute, while it
    actually shoud! **Use with great care!**

    Notes
    -----
    * Meant to be used as a decorator.
    * The wrapped function may only have positional arguments.
    * Works with functions and methods.

    Examples
    --------
    >>> @cache_by_id
    ... def f(x, y):
    ...     print("Calling f")
    ...     return x, y
    >>> f(1, 2)
    Calling f
    (1, 2)
    >>> f(1, 2)
    (1, 2)
    >>> f(1, 1)
    Calling f
    (1, 1)
    >>> f(1, 1)
    (1, 1)
    """

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self._cached_value = None
        self._cached_index = None

    def __call__(self, *args):
        index = tuple(id(arg) for arg in args)

        if index != self._cached_index:
            self._cached_index = index
            self._cached_value = self.func(*args)

        return self._cached_value

    def __get__(self, instance, owner):
        # See https://stackoverflow.com/questions/30104047 for full explanation
        return functools.partial(self.__call__, instance)


class LoggingContext(object):
    """
    This context manager allows for a temporary override of logger settings.
    """

    # from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


class Singleton(type):
    """
    A simple singleton implementation. See [1]_ for details.

    References
    -------
    .. [1] `Creating a singleton in Python on Stack Overflow <https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python>`__.

    Examples
    --------

    .. testsetup:: singleton

       from eradiate.util.misc import Singleton

    .. doctest:: singleton

       >>> class MySingleton(metaclass=Singleton): ...
       >>> my_singleton1 = MySingleton()
       >>> my_singleton2 = MySingleton()
       >>> my_singleton1 is my_singleton2
       True

    .. testcleanup:: singleton

       del Singleton
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def camel_to_snake(name):
    # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def deduplicate(value: t.Sequence, preserve_order: bool = True) -> list:
    """
    Remove duplicates from a sequence.

    Parameters
    ---------
    value : sequence
        Sequence to remove duplicates from.

    preserve_order : bool, optional, default: True
        If ``True``, preserve item ordering. The first occurrence of duplicated
        items is kept. Setting to ``False`` may slightly improve performance.

    Returns
    -------
    list
        List of values with duplicates removed.
    """

    if preserve_order:
        return list(OrderedDict.fromkeys(value))

    else:
        return list(set(value))


def deduplicate_sorted(value: t.Sequence, cmp: t.Callable | None = None) -> list:
    if cmp is None:
        cmp = lambda x, y: x == y  # noqa: E731

    result = [value[0]]

    for i in range(1, len(value)):
        if not cmp(value[i], value[i - 1]):
            result.append(value[i])

    return result


def flatten(d: t.Mapping, sep: str = ".", name: str = "") -> dict:
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to be flattened.

    name : str, optional, default: ""
        Path to the parent dictionary. By default, no parent name is defined.

    sep : str, optional, default: "."
        Flattened dict key separator.

    Returns
    -------
    dict
        A flattened copy of `d`.

    See Also
    --------
    :func:`.nest`, :func:`.set_nested`
    """
    result = {}

    for k, v in d.items():
        full_key = k if not name else f"{name}{sep}{k}"
        if isinstance(v, dict):
            result.update(flatten(v, sep=sep, name=full_key))
        else:
            result[full_key] = v

    return result


def fullname(obj: t.Any) -> str:
    """
    Get the fully qualified name of `obj`. Aliases will be dereferenced.
    """
    cls = get_class_that_defined_method(obj)

    if cls is None:
        return f"{obj.__module__}.{obj.__qualname__}"

    # else: # (it's a method)
    return f"{cls.__module__}.{obj.__qualname__}"


def get_class_that_defined_method(meth: t.Any) -> type:
    """
    Get the class which defined a method, if relevant. Otherwise, return
    ``None``.
    """
    # See https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)

    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)  # fallback to __qualname__ parsing

    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
            None,
        )
        if isinstance(cls, type):
            return cls

    return getattr(meth, "__objclass__", None)  # handle special descriptor objects


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
    Simple sort key natural order for string sorting. See [2]_ for details.

    See Also
    --------
    `Sorting HOWTO <https://docs.python.org/3/howto/sorting.html>`__

    References
    ----------
    .. [2] `Natural sorting on Stack Overflow <https://stackoverflow.com/a/11150413/3645374>`__.
    """
    return tuple(
        map(
            lambda text: int(text) if text.isdigit() else text.lower(),
            re.split("([0-9]+)", x),
        )
    )


def natsorted(l):  # noqa
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


def nest(d: t.Mapping, sep: str = ".") -> dict:
    """
    Turn a flat dictionary into a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to be unflattened.

    sep : str, optional, default: "."
        Flattened dict key separator.

    Returns
    -------
    dict
        A nested copy of `d`.

    See Also
    --------
    :func:`.flatten`, :func:`.set_nested`
    """
    result = {}

    for key, value in d.items():
        set_nested(result, key, value, sep)

    return result


def onedict_value(d: t.Mapping) -> t.Any:
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

    .. testsetup:: onedict_value

       from eradiate.util.misc import onedict_value

    .. doctest:: onedict_value

       >>> onedict_value({"foo": "bar"})
       'bar'

    .. testcleanup:: onedict_value

       del onedict_value
    """

    if len(d) != 1:
        raise ValueError(f"dictionary has wrong length (expected 1, got {len(d)}")

    return next(iter(d.values()))


def round_to_multiple(number, multiple, direction="nearest"):
    if direction == "nearest":
        return multiple * round(number / multiple)
    elif direction == "up":
        return multiple * np.ceil(number / multiple)
    elif direction == "down":
        return multiple * np.floor(number / multiple)
    else:
        return multiple * round(number / multiple)


def set_nested(d: t.Mapping, path: str, value: t.Any, sep: str = ".") -> None:
    """
    Set values in a nested dictionary using a flat path.

    Parameters
    ----------
    d : dict
        Dictionary to operate on.

    path : str
        Path to the value to be set.

    value
        Value to which `path` is to be set.

    sep : str, optional, default: "."
        Separator used to decompose `path`.

    See Also
    --------
    :func:`.flatten`, :func:`.nest`
    """
    *path, last = path.split(sep)
    for bit in path:
        d = d.setdefault(bit, {})
    d[last] = value


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


@functools.singledispatch
def summary_repr(value):
    """
    Return a summarized repr for `value`.
    """
    return repr(value)


@summary_repr.register
def _(ds: xr.Dataset):
    extra_info = {}

    try:
        extra_info["source"] = repr(ds.encoding["source"])
    except KeyError:
        pass

    desc = ", ".join([f"{key}={value}" for key, value in extra_info.items()])

    if desc:
        desc = " | " + desc

    return f"<xarray.Dataset{desc}>"


@summary_repr.register
def _(da: xr.DataArray):
    extra_info = {}

    try:
        extra_info["name"] = repr(da.name)
    except AttributeError:
        pass

    extra_info["dims"] = repr(list(da.dims))

    try:
        extra_info["source"] = repr(da.encoding["source"])
    except KeyError:
        pass

    desc = ", ".join([f"{key}={value}" for key, value in extra_info.items()])

    if desc:
        desc = " | " + desc

    return f"<xarray.DataArray{desc}>"


@summary_repr.register
def _(x: pint.Quantity):
    """
    Return a brief summary representation of a Pint quantity.
    """
    return f"{summary_repr_vector(x.m)} {x.u:~}"


def summary_repr_vector(a: np.ndarray, edgeitems: int = 4):
    """
    Return a brief summary representation of a Numpy vector.
    """
    size = len(a)
    if size > edgeitems * 2 + 1:
        return (
            f"[{np.array2string(a[:edgeitems]).strip('[]')}"
            " ... "
            f"{np.array2string(a[size - edgeitems :]).strip('[]')}]"
        )
    else:
        return np.array2string(a)


def find_runs(
    x: npt.ArrayLike,
) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Find runs of consecutive items in an array.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    tuple(array-like, array-like, array-like)
        Run values, run starts, run lengths.

    Notes
    -----
    Credit: Alistair Miles
    Source: https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


class MultiGenerator:
    """
    This generator aggregates several generators and makes sure that items that
    have already been served are not repeated.
    """

    def __init__(self, generators):
        self.generators = generators
        self._i_generator = 0
        self._current_iterator = iter(self.generators[self._i_generator])
        self._visited = set()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = next(self._current_iterator)
            if result not in self._visited:
                self._visited.add(result)
                return result
            else:
                return self.__next__()
        except StopIteration:
            if self._i_generator >= len(self.generators) - 1:
                raise
            else:
                self._i_generator += 1
                self._current_iterator = iter(self.generators[self._i_generator])
                return self.__next__()


def dirsize(path: PathLike) -> int:
    """
    Compute the recursive size of a directory in bytes, not following symlinks.

    Parameters
    ----------
    path : path-like
        Path to the directory to compute size for.

    Returns
    -------
    int
        Total size of the directory in bytes.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    NotADirectoryError
        If the path is not a directory.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    total_size = 0

    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)

        # Add size of all files in current directory
        for file in files:
            file_path = root_path / file
            try:
                # Use lstat to not follow symlinks
                total_size += file_path.lstat().st_size
            except (OSError, FileNotFoundError):
                # Skip files that we can't access or have been deleted
                continue

        # Add size of directories themselves (directory entries)
        for dir_name in dirs:
            dir_path = root_path / dir_name
            try:
                # Only count directory entry size, not content (handled by recursion)
                if not dir_path.is_symlink():
                    total_size += dir_path.lstat().st_size
            except (OSError, FileNotFoundError):
                # Skip directories that we can't access
                continue

    return total_size
