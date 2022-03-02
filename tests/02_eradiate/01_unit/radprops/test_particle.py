import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.data import load_dataset
from eradiate.radprops import ParticleRadProfile

@pytest.fixture
def test_data_set() -> xr.Dataset:
    return load_dataset("spectra/particles/govaerts_2021-continental.nc")

def test_particle_rad_profile(mode_mono, test_data_set):
    """
    Assigns attributes.
    """
    n_layer = 12
    fractions = np.random.random(n_layer)
    z_level = np.linspace(2.0, 6.0, n_layer + 1) * ureg.km
    tau_550 = 0.2
    p = ParticleRadProfile(
        fractions=fractions,
        dataset=test_data_set,
        z_level=z_level,
        tau_550=tau_550,
    )

    assert p.dataset == test_data_set
    assert np.allclose(p.z_level, z_level)
    assert np.allclose(p.fractions, fractions)
    assert np.isclose(p.tau_550, tau_550)
