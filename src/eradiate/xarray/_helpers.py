import typing as t

import pandas as pd
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
    idx = pd.MultiIndex.from_arrays(
        (obj.vza.values.flatten(), obj.vaa.values.flatten()), names=("vza", "vaa")
    )

    # Reindex object
    result = (
        obj.drop_vars(
            # First remove the target coords, as well as the indexing dim and associated coords
            # Also remove the x film coordinate (irrelevant after unstacking)
            ("vza", "vaa", "x_index", "x")
        )
        .reindex(x_index=idx)
        .unstack()
    )  # Apply the new index and unstack

    # Reapply metadata
    result["vza"].attrs = vza_attrs
    result["vaa"].attrs = vaa_attrs

    return result
