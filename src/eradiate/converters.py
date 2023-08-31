from __future__ import annotations

__all__ = [
    "auto_or",
    "convert_absorption_data",
    "convert_thermoprops",
    "on_quantity",
    "to_dataset",
]

import os
import typing as t
from pathlib import Path

import mitsuba as mi
import numpy as np
import pint
import portion as P
import xarray as xr

import eradiate

from . import data
from .attrs import AUTO
from .data import data_store
from .data._util import locate_absorption_data
from .exceptions import UnsupportedModeError
from .typing import PathLike
from .units import to_quantity


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


def convert_thermoprops(value) -> xr.Dataset:
    """Converter for atmosphere thermophysical properties specifications."""
    import joseki

    # Dataset: do nothing
    if isinstance(value, xr.Dataset):
        return value

    # PathLike: try to load dataset
    elif isinstance(value, (os.PathLike, str)):
        path = data_store.fetch(value)
        if path.is_file():
            return joseki.load_dataset(path)
        else:
            raise ValueError(
                f"invalid path for 'thermoprops': {path} " f"(expected a file)"
            )

    # Dictionary: forward to joseki.make()
    elif isinstance(value, dict):
        return joseki.make(**value)

    # Anything else: raise error
    else:
        raise TypeError(
            f"invalid type for 'thermoprops': {type(value)} "
            f"(expected Dataset or PathLike)"
        )


def convert_absorption_data(value) -> dict[P.Interval, xr.Dataset]:
    """Converter for atmosphere absorption coefficient data."""

    # Import must be local to avoid circular imports
    from .radprops.absorption import wrange

    # dict: verify that keys are portion.Interval and values are xarray.Dataset
    if isinstance(value, dict):
        if all([isinstance(k, P.Interval) for k in value]) and all(
            [isinstance(v, xr.Dataset) for v in value.values()]
        ):
            return value
        else:
            raise ValueError(
                "All keys must be portion.Interval and all values must be "
                "xarray.Dataset"
            )

    # tuple: specifications for absorption data on the online stable data store
    elif isinstance(value, tuple):
        codename = value[0]
        wavelength_range = value[1]

        if isinstance(wavelength_range, xr.Dataset):
            srf = wavelength_range
            w = to_quantity(srf.w)
            wavelength_range = w[:: w.size - 1]

        if eradiate.mode().is_mono:
            mode = "mono"
        elif eradiate.mode().is_ckd:
            mode = "ckd"
        else:
            raise UnsupportedModeError
        paths = locate_absorption_data(
            codename=codename,
            mode=mode,
            wavelength_range=wavelength_range,
        )
        datasets = [xr.load_dataset(path) for path in paths]
        return {wrange(ds): ds for ds in datasets}

    # Dataset: compute wavelength range
    elif isinstance(value, xr.Dataset):
        return {wrange(value): value}

    # List[Dataset]: compute wavelength ranges
    elif isinstance(value, list) and all(isinstance(v, xr.Dataset) for v in value):
        return {wrange(ds): ds for ds in value}

    # Pathlike: try and load the file(s)
    elif isinstance(value, (os.PathLike, str)):
        if str(value).endswith(".nc"):  # single file
            path = data_store.fetch(value)
            ds = xr.open_dataset(path)
            return {wrange(ds): ds}

        else:  # assume 'value' is a local directory
            path = Path(value)
            files = list(path.glob("*.nc"))
            datasets = [xr.open_dataset(file) for file in files]
            return {wrange(ds): ds for ds in datasets}

    # Anything else: raise error
    else:
        raise TypeError(
            f"invalid type for 'absorption_data': {type(value)} "
            f"(expected dict or Dataset or list of Dataset or str or PathLike)"
        )
