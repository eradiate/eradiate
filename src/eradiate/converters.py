from __future__ import annotations

__all__ = [
    "auto_or",
    "on_quantity",
    "to_dataset",
]

import os
import typing as t

import mitsuba as mi
import numpy as np
import pint
import xarray as xr

from . import data
from .attrs import AUTO
from .typing import PathLike


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


def to_dataset(
    load_from_id: t.Callable[[str], xr.Dataset] | None = None,
) -> t.Callable[[xr.Dataset | PathLike], xr.Dataset]:
    """
    Generates a converter that converts a value to a :class:`xarray.Dataset`.

    Parameters
    ----------
    load_from_id : callable, optional
        A callable with the signature ``f(x: str) -> Dataset`` used to
        interpret dataset identifiers.
        Set this parameter to handle dataset identifiers.
        If unset, dataset identifiers are not supported.

    Returns
    -------
    A dataset converter.

    Notes
    -----
    The conversion logic is as follows:

    1. If the value is an xarray dataset, it is returned directly.
    2. If the value is a path-like object ending with the ``.nc`` extension, the
       converter tries to load a dataset from that location, first locally, then
       (should that fail) from the Eradiate data store.
    3. If the value is a string and ``load_from_id`` is not ``None``, it is
        interpreted as a dataset identifier and ``load_from_id(value)`` is
        returned.
    4. Otherwise, a :class:`ValueError` is raised.

    Examples
    --------
    A converter with basic dataset identifier interpretation (the passed
    callable may implement more complex logic, *e.g.* with identifier
    fallback substitution):

    >>> aerosol_converter = to_dataset(
    ...     lambda x: data.load_dataset(f"spectra/particles/{x}.nc")
    ... )

    A converter without dataset identifier interpretation:

    >>> aerosol_converter = to_dataset()
    """

    def converter(value: xr.Dataset | PathLike) -> xr.Dataset:
        if isinstance(value, xr.Dataset):
            return value

        # Path (local or remote)
        if str(value).endswith(".nc"):
            # Try and open a file if it is directly referenced
            if os.path.isfile(value):
                return xr.load_dataset(value)

            # Try and serve the file from the data store
            return data.load_dataset(value)

        # Identifier for a dataset in the data store
        if isinstance(value, str) and load_from_id is not None:
            return load_from_id(value)

        # Abnormal state
        # Reference must be provided as a Dataset, a path-like or a str
        raise ValueError(f"Cannot convert value '{value}'")

    return converter


def to_mi_scalar_transform(value):
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
