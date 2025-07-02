from __future__ import annotations

__all__ = [
    "auto_or",
    "convert_thermoprops",
    "on_quantity",
    "passthrough",
    "passthrough_type",
    "resolve_keyword",
    "resolve_path",
    "to_mi_scalar_transform",
]

import os
from pathlib import Path
from typing import Any, Callable

import attrs
import mitsuba as mi
import numpy as np
import pint
import xarray as xr

from .attrs import AUTO
from .data import fresolver
from .exceptions import DataError
from .typing import PathLike


def on_quantity(
    wrapped_converter: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """
    Apply a converter to the magnitude of a :class:`pint.Quantity`.

    Parameters
    ----------
    wrapped_converter : callable
        The converter which will be applied to the magnitude of a
        :class:`pint.Quantity`.

    Returns
    -------
    callable
    """

    def f(value: Any) -> Any:
        if isinstance(value, pint.Quantity):
            return wrapped_converter(value.magnitude) * value.units
        else:
            return wrapped_converter(value)

    return f


def auto_or(
    wrapped_converter: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """
    A converter that allows an attribute to be set to :data:`.AUTO`.

    Parameters
    ----------
    wrapped_converter : callable
        The converter that is used for non-:data:`.AUTO` values.

    Returns
    -------
    callable
    """

    def f(value):
        if value is AUTO:
            return value

        return wrapped_converter(value)

    return f


def passthrough(predicate: Callable[[Any], bool]) -> Callable[[Any], Any]:
    """
    Pass through values for which ``predicate`` returns ``True``; otherwise,
    apply wrapped converter.

    See Also
    --------
    :func:`.passthrough_type`
    """

    def wrapped_converter(converter: Callable | attrs.Converter) -> Any:
        if isinstance(
            converter, attrs.Converter
        ):  # See https://github.com/python-attrs/attrs/pull/1372

            def passthrough_converter(val, inst, field):
                return val if predicate(val) else converter(val, inst, field)

            return attrs.Converter(
                passthrough_converter, takes_self=True, takes_field=True
            )

        else:

            def passthrough_converter(val):
                return val if predicate(val) else converter(val)

            return passthrough_converter

    return wrapped_converter


def passthrough_type(types: type | tuple[type, ...]) -> Callable:
    """
    Pass through values of a specified type; otherwise, apply wrapped converter.

    See Also
    --------
    :func:`.passthrough`
    """
    return passthrough(lambda x: isinstance(x, types))


def resolve_keyword(path_forming_func: Callable[[Any], PathLike]) -> Callable:
    """
    Attempt resolving a keyword into a path constructed from a keyword by the
    ``path_forming_func`` parameter.

    If the generated path points to a file, the path is returned; otherwise,
    ``value`` is returned without modification.

    Parameters
    ----------
    path_forming_func : callable
        A callable with signature ``f(x: str) -> Path`` that constructs relative
        or absolute paths from keywords. Relative paths are then resolved by the
        file resolver.
    """

    def resolve_keyword_converter(value):
        path = fresolver.resolve(path_forming_func(value))
        if path.is_file():
            return path
        else:
            return value

    return resolve_keyword_converter


def resolve_path(value: PathLike) -> Path:
    """
    Resolve a file path with the file resolver. The current working directory is
    included in the path lookup.
    """
    return fresolver.resolve(value, cwd=True)


def load_dataset(value: PathLike) -> xr.Dataset:
    """
    Attempt loading a dataset given a path. If the path is relative, it is
    resolved by the file resolver first.

    Parameters
    ----------
    value
        Path to the targeted dataset.

    Raises
    ------
    DataError
        If the file could not be loaded.
    """
    path = resolve_path(value)
    try:
        return xr.load_dataset(path)
    except Exception as e:
        raise DataError(f"could not load dataset '{value}'") from e


def to_mi_scalar_transform(value: Any):
    """
    Convert an array-like value to a :class:`mitsuba.ScalarTransform4f`.
    If `value` is a Numpy array, it is used to initialize a
    :class:`mitsuba.ScalarTransform4f` without copy; if it is a list, a Numpy
    array is first created from it. Otherwise, `value` is forwarded without
    change.
    """
    if isinstance(value, np.ndarray):
        return mi.ScalarTransform4f(value)

    elif isinstance(value, list):
        return mi.ScalarTransform4f(np.array(value))

    else:
        return value


def convert_thermoprops(value: Any) -> xr.Dataset:
    """Converter for atmosphere thermophysical properties specifications."""
    import joseki

    # Dataset: do nothing
    if isinstance(value, xr.Dataset):
        return value

    # PathLike: try to load dataset
    if isinstance(value, (os.PathLike, str)):
        path = fresolver.resolve(value)
        if path.is_file():
            return joseki.load_dataset(path)
        else:
            raise ValueError(
                f"invalid path for 'thermoprops': {path} (expected a file)"
            )

    # Dictionary: forward to joseki.make()
    if isinstance(value, dict):
        return joseki.make(**value)

    # Anything else: raise error
    raise TypeError(
        f"invalid type for 'thermoprops': {type(value)} (expected Dataset or path-like)"
    )
