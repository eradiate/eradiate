from typing import List

import xarray as xr


def weighted_mean(
    data_arrays: List[xr.DataArray], weights: List[xr.DataArray]
) -> xr.DataArray:
    """Compute the weighted mean.

    Parameter ``data_arrays`` (list of :class:`~xarray.DataArray`):
        Data arrays of which to compute the weighted mean.

    Parameter ``weights`` (list of :class:`~xarray.DataArray`):
        Weights data.

    Returns → :class:`~xarray.DataArray`:
        Weighted mean data array.
    """
    return sum([da * w for da, w in zip(data_arrays, weights)]) / sum(weights)
