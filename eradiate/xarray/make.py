import xarray as xr


def make_dataarray(data, coords=None, dims=None, var_spec=None):
    """Create a :class:`~xarray.DataArray` with default metadata.

    Parameter ``data``:
        Data forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``coords``:
        Coordinates forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``dims``:
        Dimension names forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``var_spec`` (:class:`.VarSpec` or None):
        If not ``None``, data variable specification used to apply default
        metadata.
    """

    dataarray = xr.DataArray(data, coords=coords, dims=dims)
    if var_spec is not None:
        dataarray.ert.normalize_metadata(var_spec)

    return dataarray
