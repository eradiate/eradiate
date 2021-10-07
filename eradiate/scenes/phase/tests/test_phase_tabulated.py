import pathlib

import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.phase._tabulated import TabulatedPhaseFunction


@pytest.fixture
def dataset():
    result = xr.open_dataset(
        path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc")
    ).load()
    result.close()
    return result


def test_phase_tabulated_basic(modes_all, dataset):
    # The phase function can be constructed
    phase = TabulatedPhaseFunction(data=dataset.phase)

    # Its kernel dict can be generated
    ctx = KernelDictContext()
    kernel_dict = phase.kernel_dict(ctx)

    # The kernel dict can be loaded
    assert kernel_dict.load()


def test_phase_tabulated_order(mode_mono, tmpdir):
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
