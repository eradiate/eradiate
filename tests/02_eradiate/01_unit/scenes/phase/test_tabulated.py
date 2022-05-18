import numpy as np
import pytest
import xarray as xr

from eradiate import data
from eradiate import path_resolver
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.phase._tabulated import TabulatedPhaseFunction


@pytest.fixture
def regular_mu() -> xr.DataArray:
    """Phase function data with a regular mu grid."""
    result = data.load_dataset("tests/spectra/particles/random-regular_mu.nc")
    return result.phase


@pytest.fixture
def irregular_mu() -> xr.DataArray:
    """Phase function data with an irregular mu grid."""
    result = data.load_dataset("tests/spectra/particles/random-irregular_mu.nc")
    return result.phase


def test_tabulated_basic(modes_all, regular_mu):
    # The phase function can be constructed
    phase = TabulatedPhaseFunction(data=regular_mu)

    # Its kernel dict can be generated
    ctx = KernelDictContext()
    kernel_dict = phase.kernel_dict(ctx)

    # The kernel dict can be loaded
    assert kernel_dict.load()


def test_tabulated_order(mode_mono, tmpdir):
    """
    TabulatedPhaseFunction returns phase function values by increasing order of
    scattering angle cosine values, regardless how its input is ordered.
    """

    def make_da(mu, phase):
        return xr.DataArray(
            phase[:, :, np.newaxis, np.newaxis],
            coords={
                "w": ("w", [240.0, 2800.0], dict(units="nm")),
                "mu": ("mu", mu),
                "i": ("i", [0]),
                "j": ("j", [0]),
            },
        )

    da_mu_increasing = make_da(
        mu=np.linspace(-1, 1, 3), phase=np.array([np.arange(1, 4), np.arange(1, 4)])
    )

    da_mu_decreasing = make_da(
        mu=np.linspace(1, -1, 3),
        phase=np.array([np.arange(3, 0, -1), np.arange(3, 0, -1)]),
    )

    spectral_ctx = SpectralContext.new()

    layer_mu_increasing = TabulatedPhaseFunction(data=da_mu_increasing)
    phase_mu_increasing = layer_mu_increasing.eval(spectral_ctx)

    layer_mu_decreasing = TabulatedPhaseFunction(data=da_mu_decreasing)
    phase_mu_decreasing = layer_mu_decreasing.eval(spectral_ctx)

    assert np.all(phase_mu_increasing == phase_mu_decreasing)


def test_tabulated_eval_regular_mu(modes_all, regular_mu):
    """
    Phase function data with regular mu grid is not interpolated along mu.
    """
    tabphase = TabulatedPhaseFunction(data=regular_mu)
    spectral_ctx = SpectralContext.new()
    phase = tabphase.eval(spectral_ctx)
    assert phase.size == regular_mu.mu.size


def test_tabulated_eval_irregular_mu(modes_all, irregular_mu):
    """
    Phase function data with irregular mu grid is interpolated on a regular
    grid that has enough points to sample the input data.
    """
    tabphase = TabulatedPhaseFunction(data=irregular_mu)
    spectral_ctx = SpectralContext.new()
    phase = tabphase.eval(spectral_ctx)
    dmu = irregular_mu.mu.diff(dim="mu").values.min()
    nmu = int(np.ceil(2.0 / dmu)) + 1
    assert phase.size == nmu
