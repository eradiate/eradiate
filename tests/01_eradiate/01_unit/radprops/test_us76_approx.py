import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.radprops import US76ApproxRadProfile, ZGrid
from eradiate.spectral.index import SpectralIndex


@pytest.fixture
def absorption_dataset_550nm():
    return "spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"


def test_us76_approx_rad_profile(mode_mono, absorption_dataset_550nm):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    p = US76ApproxRadProfile(absorption_dataset=absorption_dataset_550nm)

    si = SpectralIndex.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}")(si, p.zgrid)
        assert isinstance(x, ureg.Quantity)


def test_us76_approx_rad_profile_levels(mode_mono, absorption_dataset_550nm):
    """
    Collision coefficients' shape match altitude levels shape.
    """
    p = US76ApproxRadProfile(
        thermoprops=dict(levels=ureg.Quantity(np.linspace(0, 120, 121), "km")),
        absorption_dataset=absorption_dataset_550nm,
    )

    si = SpectralIndex.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}")(si, p.zgrid)
        assert x.shape == (120,)


def test_us76_approx_rad_profile_has_absorption_default(mode_mono):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    p = US76ApproxRadProfile(
        absorption_dataset="spectra/absorption/us76_u86_4/us76_u86_4-spectra-6000_7000.nc",
    )
    assert p.has_absorption
    si = SpectralIndex.new(w=1650 * ureg.nm)
    ds = p.eval_dataset(si, p.zgrid)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_true(mode_mono):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    p = US76ApproxRadProfile(
        absorption_dataset="spectra/absorption/us76_u86_4/us76_u86_4-spectra-6000_7000.nc",
        has_absorption=True,
    )
    assert p.has_absorption
    si = SpectralIndex.new(w=1650.0 * ureg.nm)
    ds = p.eval_dataset(si, p.zgrid)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_false(
    mode_mono,
    absorption_dataset_550nm,
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    p = US76ApproxRadProfile(
        absorption_dataset=absorption_dataset_550nm,
        has_absorption=False,
    )
    assert not p.has_absorption
    si = SpectralIndex.new(w=1650.0 * ureg.nm)
    ds = p.eval_dataset(si, p.zgrid)
    assert (ds.sigma_a.values == 0.0).all()


def test_us76_approx_rad_profile_has_scattering_default(
    mode_mono,
    absorption_dataset_550nm,
):
    """
    Default value for 'has_scattering' is True, hence the scattering
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    p = US76ApproxRadProfile(absorption_dataset=absorption_dataset_550nm)
    assert p.has_scattering
    si = SpectralIndex.new(w=550.0 * ureg.nm)
    ds = p.eval_dataset(si, p.zgrid)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_true(
    mode_mono,
    absorption_dataset_550nm,
):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    p = US76ApproxRadProfile(
        has_scattering=True,
        absorption_dataset=absorption_dataset_550nm,
    )
    assert p.has_scattering
    si = SpectralIndex.new(w=550 * ureg.nm)
    ds = p.eval_dataset(si, p.zgrid)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_false(
    mode_mono,
    absorption_dataset_550nm,
):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    p = US76ApproxRadProfile(
        has_scattering=False,
        absorption_dataset=absorption_dataset_550nm,
    )
    assert not p.has_scattering
    si = SpectralIndex.new(w=550.0 * ureg.nm)
    ds = p.eval_dataset(si, p.zgrid)
    assert (ds.sigma_s.values == 0.0).all()


def test_us76_approx_eval_zgrid(mode_mono, absorption_dataset_550nm):
    """
    Evaluation on an arbitrary altitude grid works.
    """
    p = US76ApproxRadProfile(absorption_dataset=absorption_dataset_550nm)
    si = SpectralIndex.new(w=550.0 * ureg.nm)
    zgrid = ZGrid(levels=np.linspace(0, 100, 1001) * ureg.km)
    ds = p.eval_dataset(si, zgrid)
    assert len(ds.z_layer) == len(zgrid.layers)
