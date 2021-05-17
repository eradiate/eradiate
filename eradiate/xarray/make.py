__all__ = ["make_dataarray"]

from typing import Optional

import xarray
import xarray as xr

from eradiate.xarray.metadata import VarSpec


def make_dataarray(
    data, coords=None, dims=None, var_spec: Optional[VarSpec] = None
) -> xarray.DataArray:
    """
    Create a :class:`~xarray.DataArray` with default metadata.

    Parameter ``data``:
        Data forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``coords``:
        Coordinates forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``dims``:
        Dimension names forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``var_spec`` (:class:`.VarSpec` or None):
        If not ``None``, data variable specification used to apply default
        metadata.

    Returns → :class:`xarray.DataArray`:
        Created :class:`~xarray.DataArray`.
    """

    dataarray = xr.DataArray(data, coords=coords, dims=dims)
    if var_spec is not None:
        dataarray.ert.normalize_metadata(var_spec)

    return dataarray
