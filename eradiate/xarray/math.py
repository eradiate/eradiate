from typing import List

import numpy as np
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


def ratios(data_arrays: List[xr.DataArray]) -> xr.DataArray:
    """Compute the ratios of data arrays.

    Provided all data arrays give the value of a quantity :math:`q`, this
    function computes the ratios :math:`r_i` defined by:

    .. math::
       r_i = \\frac{q_i}{\\sum_{i}q_i}

    Parameter ``data_arrays`` (list of :class:`~xarray.DataArray`):
        Data arrays of which to compute the ratios.

    Returns → :class:`~xarray.DataArray`:
        Ratios.
    """

    da_sum = sum([da for da in data_arrays])

    ratios = []
    for i, da in enumerate(data_arrays):
        ratio = da / da_sum
        tmp = xr.DataArray(ratio.data[np.newaxis, :], dims=["i"] + list(ratio.dims))
        ratios.append(tmp.assign_coords(ratio.coords).assign_coords({"i": [i]}))

    return xr.concat(ratios, dim="i")
