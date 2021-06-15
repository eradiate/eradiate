import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops import (
    AFGL1986RadProfile,
    ArrayRadProfile,
    RadProfileFactory,
    US76ApproxRadProfile,
)
from eradiate.thermoprops.util import (
    compute_column_number_density,
    compute_number_density_at_surface,
)

# ------------------------------------------------------------------------------
#                       Radiative profile factory
# ------------------------------------------------------------------------------


def test_rad_props_profile_factory(mode_mono):
    """Creates a RadProfile object."""
    p = RadProfileFactory.create(
        {
            "type": "array",
            "levels": [0, 1, 2, 3],
            "albedo_values": [[[0, 1, 2, 3]]],
            "sigma_t_values": [[[0, 1, 2, 3]]],
        }
    )
    assert isinstance(p, ArrayRadProfile)


# ------------------------------------------------------------------------------
#                       Array radiative profile
# ------------------------------------------------------------------------------


def test_array_rad_props_profile(mode_mono):
    """Attributes match specified input values and are of type Quantity."""
    levels = ureg.Quantity(np.linspace(0, 100, 12), "km")
    albedo_values = ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless)
    sigma_t_values = ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1")
    p = ArrayRadProfile(
        levels=levels,
        albedo_values=albedo_values.reshape(1, 1, len(levels) - 1),
        sigma_t_values=sigma_t_values.reshape(1, 1, len(levels) - 1),
    )
    assert isinstance(p.levels, ureg.Quantity)
    assert isinstance(p.sigma_a(), ureg.Quantity)
    assert isinstance(p.sigma_s(), ureg.Quantity)
    assert np.allclose(p.levels, levels)
    assert np.allclose(p.albedo(), albedo_values)
    assert np.allclose(p.sigma_t(), sigma_t_values)


def test_array_rad_props_profile_to_dataset(mode_mono):
    """Returns a xarray.Dataset object."""
    levels = ureg.Quantity(np.linspace(0, 100, 12), "km")
    albedo_values = ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless)
    sigma_t_values = ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1")
    p = ArrayRadProfile(
        levels=levels,
        albedo_values=albedo_values.reshape(1, 1, len(levels) - 1),
        sigma_t_values=sigma_t_values.reshape(1, 1, len(levels) - 1),
    )
    spectral_ctx = SpectralContext.new()
    assert isinstance(p.to_dataset(spectral_ctx), xr.Dataset)


def test_array_rad_props_profile_incompatible_albedo_values_and_sigma_t_values(
    mode_mono,
):
    """Raises when albedo_values and sigma_t_values are incompatible."""
    with pytest.raises(ValueError):
        ArrayRadProfile(
            levels=ureg.Quantity(np.linspace(0, 100, 12), "km"),
            albedo_values=np.linspace(0.0, 1.0, 11).reshape(1, 1, 11),
            sigma_t_values=np.linspace(0.0, 1e-5, 10).reshape(1, 1, 10),
        )


# ------------------------------------------------------------------------------
#                       US76 approximated radiative profile
# ------------------------------------------------------------------------------


@pytest.fixture
def us76_approx_test_absorption_data_set():
    return path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def _assert_attribtues_have_right_type_and_shape(p, spectral_ctx, shape):
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(p, field)(spectral_ctx)
        assert isinstance(x, ureg.Quantity)
        assert x.shape == shape


def test_us76_approx_rad_profile(mode_mono, us76_approx_test_absorption_data_set):
    """Default constructor assigns attributes to specified input values."""
    spectral_ctx = SpectralContext.new()
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)
    _assert_attribtues_have_right_type_and_shape(p, spectral_ctx, (1, 1, 86))


def test_us76_approx_rad_profile(mode_mono, us76_approx_test_absorption_data_set):
    """Custom altitude levels are applied."""
    spectral_ctx = SpectralContext.new()
    p = US76ApproxRadProfile(
        levels=ureg.Quantity(np.linspace(0, 120, 121), "km"),
        absorption_data_set=us76_approx_test_absorption_data_set,
    )
    _assert_attribtues_have_right_type_and_shape(p, spectral_ctx, (1, 1, 120))


def test_us76_approx_rad_profile_eval_sigma_a(
    mode_mono, us76_approx_test_absorption_data_set
):
    spectral_ctx = SpectralContext.new()
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)
    assert isinstance(p.eval_sigma_a(spectral_ctx), ureg.Quantity)


def test_us76_approx_rad_profile_has_absorption_default(
    mode_mono, us76_approx_test_absorption_data_set
):
    """Default 'has_absorption' triggers absorption computation which produces
    non-zero absorption coefficient values at a wavelength of 1650 nm."""
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_true(
    mode_mono, us76_approx_test_absorption_data_set
):
    """Setting 'has_absorption' to True triggers the computation of the
    absorption coefficient which is not zero everywhere at 1650 nm."""
    p = US76ApproxRadProfile(
        has_absorption=True, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_us76_approx_rad_profile_has_absorption_false(
    mode_mono, us76_approx_test_absorption_data_set
):
    """Setting 'has_absorption' to False disables the computation of the
    absorption coefficient and produces zero absorption coefficient values."""
    p = US76ApproxRadProfile(
        has_absorption=False, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert not p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_a.values == 0.0).all()


def test_us76_approx_rad_profile_has_scattering_default(
    mode_mono, us76_approx_test_absorption_data_set
):
    """Using the default 'has_scattering' value enables the scattering coefficient
    computation which produces non zero scattering coefficient values at 550 nm."""
    p = US76ApproxRadProfile(absorption_data_set=us76_approx_test_absorption_data_set)
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_true(
    mode_mono, us76_approx_test_absorption_data_set
):
    """Setting 'has_scattering' to True, enables the scattering coefficient
    computation which produces non zero scattering coefficient values at 550 nm."""
    p = US76ApproxRadProfile(
        has_scattering=True, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_us76_approx_rad_profile_has_scattering_false(
    mode_mono, us76_approx_test_absorption_data_set
):
    """Setting 'has_scattering' to False, disables the scattering coefficient
    computation which produces zero scattering coefficient values at 550 nm."""
    p = US76ApproxRadProfile(
        has_scattering=False, absorption_data_set=us76_approx_test_absorption_data_set
    )
    assert not p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_s.values == 0.0).all()


# ------------------------------------------------------------------------------
#                       AFGL 1986 radiative profile
# ------------------------------------------------------------------------------


@pytest.fixture
def afgl1986_test_absorption_data_sets():
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
    """Default constructor assigns attributes to specified input values."""
    spectral_ctx = SpectralContext.new(wavelength=1500.0)
    # Note: in the infrared, all absorption data sets are opened

    p = AFGL1986RadProfile(absorption_data_sets=afgl1986_test_absorption_data_sets)
    _assert_attribtues_have_right_type_and_shape(p, spectral_ctx, (1, 1, 120))


def test_afgl1986_rad_profile_levels(mode_mono, afgl1986_test_absorption_data_sets):
    """Custom level altitudes are applied."""
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    # Note: in the visible, only the H2O data set is opened

    p = AFGL1986RadProfile(
        levels=ureg.Quantity(np.linspace(0, 100, 101), "km"),
        absorption_data_sets=afgl1986_test_absorption_data_sets,
    )
    _assert_attribtues_have_right_type_and_shape(p, spectral_ctx, (1, 1, 100))


def test_afgl1986_rad_profile_concentrations(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Custom concentrations are applied."""
    concentrations = {
        "H2O": ureg.Quantity(5e23, "m^-2"),  # column number density in S.I. units
        "O3": ureg.Quantity(
            0.5, "dobson_unit"
        ),  # column number density in exotic units
        "CH4": ureg.Quantity(4e19, "m^-3"),  # number density at the surface
        "CO2": ureg.Quantity(400e-6, ""),  # mixing ratio at the surface
    }
    p = AFGL1986RadProfile(
        concentrations=concentrations,
        absorption_data_sets=afgl1986_test_absorption_data_sets,
    )

    thermoprops = p.eval_thermoprops_profile()
    column_amount_H2O = compute_column_number_density(thermoprops, "H2O")
    column_amount_O3 = compute_column_number_density(thermoprops, "O3")
    surface_amount_CH4 = compute_number_density_at_surface(thermoprops, "CH4")
    surface_amount_CO2 = thermoprops.mr.sel(species="CO2").values[0]

    assert np.isclose(column_amount_H2O, concentrations["H2O"], rtol=1e-9)
    assert np.isclose(column_amount_O3, concentrations["O3"], rtol=1e-9)
    assert np.isclose(surface_amount_CO2, concentrations["CO2"], rtol=1e-9)
    assert np.isclose(surface_amount_CH4, concentrations["CH4"], rtol=1e-9)


def test_afgl1986_rad_profile_concentrations_invalid(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Raises when a concentration value is too large."""
    p = AFGL1986RadProfile(
        concentrations={"CO2": ureg.Quantity(400, "")},
        absorption_data_sets=afgl1986_test_absorption_data_sets,
    )
    with pytest.raises(ValueError):
        p.eval_thermoprops_profile()


def test_afgl1986_rad_profile_has_absorption_default(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Default 'has_absorption' triggers absorption computation which produces
    non-zero absorption coefficient values at a wavelength of 1650 nm."""
    p = AFGL1986RadProfile(absorption_data_sets=afgl1986_test_absorption_data_sets)
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_afgl1986_rad_profile_has_absorption_true(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Setting 'has_absorption' to True triggers the computation of the
    absorption coefficient which is not zero everywhere at 1650 nm."""
    p = AFGL1986RadProfile(
        has_absorption=True, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_a.values != 0.0).any()


def test_afgl1986_rad_profile_has_absorption_false(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Setting 'has_absorption' to False disables the computation of the
    absorption coefficient and produces zero absorption coefficient values."""
    p = AFGL1986RadProfile(
        has_absorption=False, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert not p.has_absorption
    spectral_ctx = SpectralContext.new(wavelength=1650.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_a.values == 0.0).all()


def test_afgl1986_rad_profile_has_scattering_default(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Using the default 'has_scattering' value enables the scattering coefficient
    computation which produces non zero scattering coefficient values at 550 nm."""
    p = AFGL1986RadProfile(absorption_data_sets=afgl1986_test_absorption_data_sets)
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl1986_rad_profile_has_scattering_true(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Setting 'has_scattering' to True, enables the scattering coefficient
    computation which produces non zero scattering coefficient values at 550 nm."""
    p = AFGL1986RadProfile(
        has_scattering=True, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl1986_rad_profile_has_scattering_false(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """Setting 'has_scattering' to False, disables the scattering coefficient
    computation which produces zero scattering coefficient values at 550 nm."""
    p = AFGL1986RadProfile(
        has_scattering=False, absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    assert not p.has_scattering
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ds = p.to_dataset(spectral_ctx)
    assert (ds.sigma_s.values == 0.0).all()
