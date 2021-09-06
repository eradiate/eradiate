import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops import AFGL1986RadProfile, ArrayRadProfile, US76ApproxRadProfile
from eradiate.thermoprops.afgl1986 import make_profile as make_profile_afgl1986
from eradiate.thermoprops.util import compute_column_number_density


def test_array_rad_props_profile(mode_mono):
    """
    Assigns attributes.
    """
    levels = ureg.Quantity(np.linspace(0, 100, 12), "km")
    albedo_values = ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless)
    sigma_t_values = ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1")
    p = ArrayRadProfile(
        levels=levels,
        albedo_values=albedo_values,
        sigma_t_values=sigma_t_values,
    )

    spectral_ctx = SpectralContext.new()
    assert isinstance(p.levels, ureg.Quantity)
    assert isinstance(p.eval_albedo(spectral_ctx=spectral_ctx), ureg.Quantity)
    assert isinstance(p.eval_sigma_t(spectral_ctx=spectral_ctx), ureg.Quantity)
    assert isinstance(p.eval_sigma_a(spectral_ctx=spectral_ctx), ureg.Quantity)
    assert isinstance(p.eval_sigma_s(spectral_ctx=spectral_ctx), ureg.Quantity)
    assert np.allclose(p.levels, levels)
    assert np.allclose(p.eval_albedo(spectral_ctx=spectral_ctx), albedo_values)
    assert np.allclose(p.eval_sigma_t(spectral_ctx=spectral_ctx), sigma_t_values)


def test_array_rad_props_profile_eval_dataset(mode_mono):
    """
    Returns a data set.
    """
    p = ArrayRadProfile(
        levels=ureg.Quantity(np.linspace(0, 100, 12), "km"),
        albedo_values=ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless),
        sigma_t_values=ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1"),
    )
    spectral_ctx = SpectralContext.new()
    assert isinstance(p.eval_dataset(spectral_ctx), xr.Dataset)


def test_array_rad_props_profile_invalid_values(mode_mono):
    """
    Mismatching shapes in albedo_values and sigma_t_values arrays raise.
    """
    with pytest.raises(ValueError):
        ArrayRadProfile(
            levels=ureg.Quantity(np.linspace(0, 100, 12), "km"),
            albedo_values=np.linspace(0.0, 1.0, 11),
            sigma_t_values=np.linspace(0.0, 1e-5, 10),
        )


@pytest.fixture
def us76_approx_test_absorption_data_set():
    """
    Fixture to return the path to a test absorption data set for 'us76_approx'.
    """
    return path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_us76_approx_rad_profile(mode_mono, us76_approx_test_absorption_data_set):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)

    spectral_ctx = SpectralContext.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}")(spectral_ctx)
        assert isinstance(x, ureg.Quantity)


def test_us76_approx_rad_profile_ckd_sigma_s(mode_ckd):
    """
    Scattering coefficient evaluation methods return pint.Quantity object.
    """
    p = US76ApproxRadProfile()

    spectral_ctx = SpectralContext.new()

    assert isinstance(p.eval_sigma_s_ckd(spectral_ctx.bindex), ureg.Quantity)


def test_us76_approx_rad_profile_levels(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    Collision coefficients' shape match altitude levels shape.
    """
    p = US76ApproxRadProfile(
        thermoprops=dict(levels=ureg.Quantity(np.linspace(0, 120, 121), "km")),
        absorption_data_set=us76_approx_test_absorption_data_set,
    )

    spectral_ctx = SpectralContext.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}")(spectral_ctx)
        assert x.shape == (120,)


def test_us76_approx_rad_profile_has_absorption_default(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_true(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    p = US76ApproxRadProfile(
        has_absorption=True, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_false(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    p = US76ApproxRadProfile(
        has_absorption=False, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert not p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values == 0.0).all()


def test_us76_approx_rad_profile_has_scattering_default(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    Default value for 'has_scattering' is True, hence the scattering
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_true(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    p = US76ApproxRadProfile(
        has_scattering=True, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_false(
    mode_mono, us76_approx_test_absorption_data_set
):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    p = US76ApproxRadProfile(
        has_scattering=False, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert not p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values == 0.0).all()


@pytest.fixture
def afgl1986_test_absorption_data_sets():
    """
    Fixture to return paths to test absorption data sets for 'afgl_1986'.
    """
    return {
        "CH4": path_resolver.resolve(
            "tests/spectra/absorption/CH4-spectra-4000_11502.nc"
        ),
        "CO2": path_resolver.resolve(
            "tests/spectra/absorption/CO2-spectra-4000_14076.nc"
        ),
        "CO": path_resolver.resolve(
            "tests/spectra/absorption/CO-spectra-4000_14478.nc"
        ),
        "H2O": path_resolver.resolve(
            "tests/spectra/absorption/H2O-spectra-4000_25711.nc"
        ),
        "N2O": path_resolver.resolve(
            "tests/spectra/absorption/N2O-spectra-4000_10364.nc"
        ),
        "O2": path_resolver.resolve(
            "tests/spectra/absorption/O2-spectra-4000_17273.nc"
        ),
        "O3": path_resolver.resolve("tests/spectra/absorption/O3-spectra-4000_6997.nc"),
    }


def test_afgl1986_rad_profile_default(mode_mono, afgl1986_test_absorption_data_sets):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    p = AFGL1986RadProfile(absorption_data_sets=afgl1986_test_absorption_data_sets)

    spectral_ctx = SpectralContext.new(wavelength=1500.0)
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}")(spectral_ctx)
        assert isinstance(x, ureg.Quantity)


def test_afgl1986_rad_profile_default_ckd(mode_ckd):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    p = AFGL1986RadProfile()

    spectral_ctx = SpectralContext.new()
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}_ckd")(spectral_ctx.bindex)
        assert isinstance(x, ureg.Quantity)


def test_afgl1986_rad_profile_levels(mode_mono, afgl1986_test_absorption_data_sets):
    """
    Collision coefficients' shape match altitude levels shape.
    """
    n_layers = 101
    p = AFGL1986RadProfile(
        thermoprops=dict(
            levels=ureg.Quantity(np.linspace(0.0, 100.0, n_layers + 1), "km")
        ),
        absorption_data_sets=afgl1986_test_absorption_data_sets,
    )

    spectral_ctx = SpectralContext.new(wavelength=550.0)
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, f"eval_{field}")(spectral_ctx)
        assert x.shape == (n_layers,)


def test_afgl1986_rad_profile_concentrations(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    Absorption coefficient is twice larger when H2O concentration is doubled.
    """

    thermoprops = make_profile_afgl1986()
    column_amount_H2O = compute_column_number_density(ds=thermoprops, species="H2O")
    p1 = AFGL1986RadProfile(
        thermoprops=thermoprops, absorption_data_sets=afgl1986_test_absorption_data_sets
    )

    p2 = AFGL1986RadProfile(
        thermoprops=dict(
            concentrations={
                "H2O": 2 * column_amount_H2O,
            }
        ),
        absorption_data_sets=afgl1986_test_absorption_data_sets,
    )
    spectral_ctx = SpectralContext.new(wavelength=1500.0 * ureg.nm)

    sigma_a_initial = p1.eval_sigma_a(spectral_ctx)
    sigma_a_doubled = p2.eval_sigma_a(spectral_ctx)

    assert np.allclose(sigma_a_doubled, 2 * sigma_a_initial, rtol=1e-2)


def test_afgl1986_rad_profile_has_absorption_default(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    p = AFGL1986RadProfile(absorption_data_sets=afgl1986_test_absorption_data_sets)
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_afgl1986_rad_profile_has_absorption_true(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    p = AFGL1986RadProfile(
        has_absorption=True, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_afgl1986_rad_profile_has_absorption_false(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    p = AFGL1986RadProfile(
        has_absorption=False, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert not p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_a.values == 0.0).all()


def test_afgl1986_rad_profile_has_scattering_default(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    Default value for 'has_scattering' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    p = AFGL1986RadProfile(absorption_data_sets=afgl1986_test_absorption_data_sets)
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl1986_rad_profile_has_scattering_true(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    p = AFGL1986RadProfile(
        has_scattering=True, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl1986_rad_profile_has_scattering_false(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    p = AFGL1986RadProfile(
        has_scattering=False, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert not p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.eval_dataset(spectral_ctx)
    assert (ds.sigma_s.values == 0.0).all()
