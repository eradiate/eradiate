import typing as t

import xarray as xr

XarrayObj = t.Union[xr.DataArray, xr.Dataset]


def unstack_mdistant_grid(obj: XarrayObj) -> XarrayObj:
    """
    Reindex data produced by a :meth:`.MultiDistantMeasure.grid()` measure on
    (VAA, VZA) dimensions.

    This effectively reshapes the data, which is initially indexed on film width
    and height dimensions (the latter having a single element).

    Parameters
    ----------
    obj : DataArray or Dataset
        An xarray object produced by processing a
        :meth:`.MultiDistantMeasure.grid()` measure, indexed on film dimensions.

    Returns
    -------
    DataArray or Dataset
        An object of the same type, reindexed on VZA and VAA coordinates.
    """
    # Collect metadata
    vza_attrs = obj["vza"].attrs
    vaa_attrs = obj["vaa"].attrs

    # Build new index
    midx_values = (obj.vza.values.flatten(), obj.vaa.values.flatten())
    result = obj.drop_vars(
        ["x_index", "x"]
    )  # Remove the x film coordinate (irrelevant after unstacking)
    result = result.set_xindex(
        ["vza", "vaa"], tuples=midx_values
    )  # Reindex the x_index dimension with a multi-index using the vza and vaa variables
    result = result.unstack()  # Unstack the multi-level index

    # Reapply metadata
    result["vza"].attrs = vza_attrs
    result["vaa"].attrs = vaa_attrs

    return result
