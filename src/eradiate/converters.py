import os
import typing as t

import pint
import xarray as xr

from . import data
from .attrs import AUTO

__all__ = [
    "auto_or",
    "load_dataset",
    "on_quantity",
]


def on_quantity(
    wrapped_converter: t.Callable[[t.Any], t.Any]
) -> t.Callable[[t.Any], t.Any]:
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

    def f(value: t.Any) -> t.Any:
        if isinstance(value, pint.Quantity):
            return wrapped_converter(value.magnitude) * value.units
        else:
            return wrapped_converter(value)

    return f


def auto_or(
    wrapped_converter: t.Callable[[t.Any], t.Any]
) -> t.Callable[[t.Any], t.Any]:
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


def load_dataset(value: t.Any) -> t.Any:
    """
    Try to load an xarray dataset from the passed value:

    * if `value` is a string or path-like object, it attempts to load
      a dataset from that location;
    * if the previous step fails, it tries to serve it from the data store;
    * if `value` is an xarray dataset, it is returned directly;
    * otherwise, a :class:`ValueError` is raised.
    """
    if isinstance(value, (str, os.PathLike, bytes)):
        # Try to open a file if it is directly referenced
        if os.path.isfile(value):
            return xr.load_dataset(value)

        # Try to serve the file from the data store
        return data.load_dataset(value)

    elif isinstance(value, xr.Dataset):
        return value

    else:
        raise ValueError(
            "Reference must be provided as a Dataset or a file path. "
            f"Got {type(value).__name__}"
        )
