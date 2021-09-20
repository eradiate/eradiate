__all__ = ["make_dataarray"]

import typing as t

import xarray
import xarray as xr

from eradiate.xarray.metadata import VarSpec


def make_dataarray(
    data, coords=None, dims=None, var_spec: t.Optional[VarSpec] = None
) -> xarray.DataArray:
    """
    Create a :class:`~xarray.DataArray` with default metadata.

    Parameters
    ----------
    data
        Data forwarded to the :class:`~xarray.DataArray` constructor.

    coords
        Coordinates forwarded to the :class:`~xarray.DataArray` constructor.

    dims
        Dimension names forwarded to the :class:`~xarray.DataArray` constructor.

    var_spec : :class:`.VarSpec`, optional
        If not ``None``, data variable specification used to apply default
        metadata.

    Returns
    -------
    DataArray
        Created :class:`~xarray.DataArray`.
    """

    dataarray = xr.DataArray(data, coords=coords, dims=dims)
    if var_spec is not None:
        dataarray.ert.normalize_metadata(var_spec)

    return dataarray
