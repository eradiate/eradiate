import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops import US76ApproxRadProfile, ZGrid


@pytest.fixture
def us76_approx_test_absorption_data_set():
    """
    Fixture to return the path to a test absorption data set for 'us76_approx'.
    """
    return eradiate.data.data_store.fetch(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_us76_approx_rad_profile(mode_mono, us76_approx_test_absorption_data_set):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    profile = US76ApproxRadProfile(
        absorption_data_set=us76_approx_test_absorption_data_set
    )

    spectral_ctx = SpectralContext.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(profile, f"eval_{field}")(spectral_ctx, profile.zgrid)
        assert isinstance(x, ureg.Quantity)


def test_us76_approx_rad_profile_levels(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    Collision coefficients' shape match altitude levels shape.
    """
    profile = US76ApproxRadProfile(
        thermoprops=dict(levels=ureg.Quantity(np.linspace(0, 120, 121), "km")),
        absorption_data_set=us76_approx_test_absorption_data_set,
    )

    spectral_ctx = SpectralContext.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(profile, f"eval_{field}")(spectral_ctx, profile.zgrid)
        assert x.shape == (120,)


def test_us76_approx_rad_profile_has_absorption_default(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    profile = US76ApproxRadProfile(
        absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert profile.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = profile.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_true(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    profile = US76ApproxRadProfile(
        has_absorption=True, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert profile.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = profile.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_false(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    profile = US76ApproxRadProfile(
        has_absorption=False, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert not profile.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = profile.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values == 0.0).all()


def test_us76_approx_rad_profile_has_scattering_default(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    Default value for 'has_scattering' is True, hence the scattering
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    profile = US76ApproxRadProfile(
        absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert profile.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = profile.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_true(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    profile = US76ApproxRadProfile(
        has_scattering=True, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert profile.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = profile.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_false(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    profile = US76ApproxRadProfile(
        has_scattering=False, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert not profile.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = profile.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values == 0.0).all()


def test_us76_approx_eval_zgrid(mode_mono, us76_approx_test_absorption_data_set):
    """
    Evaluation on an arbitrary altitude grid works.
    """
    profile = US76ApproxRadProfile(
        absorption_data_set=us76_approx_test_absorption_data_set
    )
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    zgrid = ZGrid(levels=np.linspace(0, 100, 1001) * ureg.km)
    ds = profile.eval_dataset(spectral_ctx, zgrid)
    assert len(ds.z_layer) == len(zgrid.layers)
