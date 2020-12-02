import numpy as np
import xarray as xr
import eradiate.data as data
from eradiate.solvers.postprocess import apply_srf


def test_apply_srf():
    da = xr.DataArray(
        np.random.random((4, 5)),
        dims=["w", "another"],
        coords={"w": 515 + 80 * np.random.random(4)})
    output_da = apply_srf(da, "sentinel-3a-slstr-s1")

    dataset = data.open(
        category="spectral_response_function",
        id="sentinel-3a-slstr-s1")
    weights = dataset.srf.interp(w=da.w.values)
    weighted = da.weighted(weights)
    weighted_sum = weighted.sum(dim="w")

    assert "another" in output_da.dims
    assert "w" not in output_da
    assert np.allclose(output_da, weighted_sum)
