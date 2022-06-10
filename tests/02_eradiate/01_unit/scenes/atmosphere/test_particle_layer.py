import numpy as np
import pytest
import xarray as xr

from eradiate import converters
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, MonoSpectralContext, SpectralContext
from eradiate.scenes.atmosphere import ParticleLayer, UniformParticleDistribution
from eradiate.scenes.measure._core import CKDMeasureSpectralConfig
from eradiate.units import to_quantity

# ------------------------------------------------------------------------------
#                              Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def test_dataset_path():
    """Test dataset path fixture."""
    return "tests/radprops/rtmom_aeronet_desert.nc"


# ------------------------------------------------------------------------------
#                                   Tests
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("wavelength", [280.0, 550.0, 1600.0, 2400.0])
def test_particle_layer_eval_mono_absorbing_only(
    mode_mono,
    absorbing_only,
    wavelength,
):
    """eval methods return expected values for an absorbing-only layer."""
    layer = ParticleLayer(dataset=absorbing_only)
    spectral_ctx = MonoSpectralContext(wavelength=wavelength)
    assert np.allclose(layer.eval_sigma_s(spectral_ctx), 0.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(spectral_ctx), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(spectral_ctx).m, 0.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_mfp(ctx).magnitude == np.inf


@pytest.mark.parametrize("wavelength", [280.0, 550.0, 1600.0, 2400.0])
def test_particle_layer_eval_mono_scattering_only(
    mode_mono, scattering_only, wavelength
):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(dataset=scattering_only)
    spectral_ctx = MonoSpectralContext(wavelength=wavelength)
    assert np.allclose(layer.eval_sigma_s(spectral_ctx), 0.2 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(spectral_ctx), 0.0 / ureg.km)
    assert np.allclose(layer.eval_albedo(spectral_ctx).m, 1.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_mfp(ctx) == 5.0 * ureg.km


@pytest.mark.parametrize("wavelength", [280.0, 550.0, 1600.0, 2400.0])
def test_particle_layer_eval_mono(
    mode_mono,
    test_particles_dataset,
    wavelength,
):
    """
    eval_* methods return expected values for a scattering and absorbing layer.
    """
    layer = ParticleLayer(
        dataset=test_particles_dataset,
        n_layers=1,
        tau_ref=1.0,
        bottom=0.0 * ureg.km,
        top=1.0 * ureg.km,
    )
    spectral_ctx = MonoSpectralContext(wavelength=wavelength)
    assert np.allclose(layer.eval_sigma_t(spectral_ctx), 1.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_s(spectral_ctx), 0.8 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(spectral_ctx), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(spectral_ctx).m, 0.8)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_mfp(ctx) == 1.25 * ureg.km


@pytest.mark.parametrize("bins", ["280", "550", "1600", "2400"])
def test_particle_layer_eval_ckd_absorbing_only(
    mode_ckd,
    absorbing_only,
    bins,
):
    """eval methods return expected values for an absorbing-only layer."""
    layer = ParticleLayer(dataset=absorbing_only)
    spectral_config = CKDMeasureSpectralConfig(bin_set="10nm", bins=bins)
    spectral_ctx = spectral_config.spectral_ctxs()[0]
    assert np.allclose(layer.eval_sigma_s(spectral_ctx), 0.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(spectral_ctx), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(spectral_ctx).m, 0.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_mfp(ctx).magnitude > 0.0


@pytest.mark.parametrize("bins", ["280", "550", "1600", "2400"])
def test_particle_layer_eval_ckd_scattering_only(
    mode_ckd,
    scattering_only,
    bins,
):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(dataset=scattering_only)
    spectral_config = CKDMeasureSpectralConfig(bin_set="10nm", bins=bins)
    spectral_ctx = spectral_config.spectral_ctxs()[0]
    assert np.all(layer.eval_sigma_s(spectral_ctx) == 0.2 / ureg.km)
    assert np.all(layer.eval_sigma_a(spectral_ctx).m == 0.0)
    assert np.all(layer.eval_albedo(spectral_ctx).m == 1.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_mfp(ctx).magnitude > 0.0


@pytest.mark.parametrize("bins", ["280", "550", "1600", "2400"])
def test_particle_layer_eval_ckd(mode_ckd, test_particles_dataset, bins):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(
        dataset=test_particles_dataset,
        n_layers=1,
        tau_ref=1.0,
        bottom=0.0 * ureg.km,
        top=1.0 * ureg.km,
    )
    spectral_config = CKDMeasureSpectralConfig(bin_set="10nm", bins=bins)
    spectral_ctx = spectral_config.spectral_ctxs()[0]
    assert np.isclose(layer.eval_sigma_t(spectral_ctx), 1.0 / ureg.km)
    assert np.isclose(layer.eval_sigma_s(spectral_ctx), 0.8 / ureg.km)
    assert np.isclose(layer.eval_sigma_a(spectral_ctx), 0.2 / ureg.km)
    assert np.isclose(layer.eval_albedo(spectral_ctx).m, 0.8)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_mfp(ctx) == 1.25 * ureg.km


def test_particle_layer_construct_basic():
    """Construction succeeds with basic parameters."""
    assert ParticleLayer(n_layers=9)


def test_particle_layer_scale(modes_all_single):
    """Scale parameter propagates to kernel dict and latter can be loaded."""
    ctx = KernelDictContext()
    d = ParticleLayer(geometry="plane_parallel", scale=2.0).kernel_dict(ctx)
    assert d["medium_atmosphere"]["scale"] == 2.0
    assert d.load()


def test_particle_layer_construct_attrs(test_dataset_path):
    """Assigns parameters to expected values."""
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    tau_ref = ureg.Quantity(0.3, "dimensionless")
    layer = ParticleLayer(
        bottom=bottom,
        top=top,
        distribution=UniformParticleDistribution(),
        tau_ref=tau_ref,
        n_layers=9,
        dataset=test_dataset_path,
    )
    assert layer.bottom == bottom
    assert layer.top == top
    assert isinstance(layer.distribution, UniformParticleDistribution)
    assert layer.tau_ref == tau_ref
    assert layer.n_layers == 9
    assert isinstance(layer.dataset, xr.Dataset)


def test_particle_layer_altitude_units():
    """Accept different units for bottom and top altitudes."""
    assert ParticleLayer(bottom=1.0 * ureg.km, top=2000.0 * ureg.m)


def test_particle_layer_invalid_bottom_top():
    """Raises when 'bottom' is larger that 'top'."""
    with pytest.raises(ValueError):
        ParticleLayer(top=1.2 * ureg.km, bottom=1.8 * ureg.km)


def test_particle_layer_invalid_tau_ref():
    """Raises when 'tau_ref' is invalid."""
    with pytest.raises(ValueError):
        ParticleLayer(
            bottom=1.2 * ureg.km,
            top=1.8 * ureg.km,
            tau_ref=-0.1 * ureg.dimensionless,
        )


def test_particle_layer_kernel_phase(modes_all_single):
    """Dictionary key is set to appropriate value."""
    atmosphere = ParticleLayer(n_layers=9)
    ctx = KernelDictContext()
    kernel_phase = atmosphere.kernel_phase(ctx)
    assert set(kernel_phase.data.keys()) == {f"phase_{atmosphere.id}"}


def test_particle_layer_kernel_dict(modes_all_single):
    """Kernel dictionary can be loaded"""
    particle_layer = ParticleLayer(geometry="plane_parallel", n_layers=9)
    ctx = KernelDictContext()
    assert particle_layer.kernel_dict(ctx).load()


def test_particle_layer_eval_radprops_format(
    modes_all_single,
    test_dataset_path,
):
    """
    Method 'eval_radprops' returns dataset with expected datavars and coords.
    """
    layer = ParticleLayer(dataset=test_dataset_path)
    spectral_ctx = SpectralContext.new()
    ds = layer.eval_radprops(spectral_ctx)
    expected_data_vars = ["sigma_t", "albedo"]
    expected_coords = ["z_layer"]
    assert all([coord in ds.coords for coord in expected_coords]) and all(
        [var in ds.data_vars for var in expected_data_vars]
    )


@pytest.mark.parametrize(
    "tau_ref",
    np.array([0.6, 1.0, 2.5]) * ureg.dimensionless,
)
def test_particle_layer_eval_radprops(mode_mono, test_dataset_path, tau_ref):
    layer = ParticleLayer(
        dataset=test_dataset_path,
        bottom=0.5 * ureg.km,  # arbitrary
        top=3.0 * ureg.km,  # arbitrary
        distribution={"type": "uniform"},
        n_layers=1,
        tau_ref=tau_ref,
    )

    # compute optical thickness at reference wavelength from layer's radprops
    # and check it matches the input tau_ref
    spectral_ctx = MonoSpectralContext(wavelength=layer.w_ref)
    radprops = layer.eval_radprops(spectral_ctx)
    delta_z = layer.height / layer.n_layers

    with xr.set_options(keep_attrs=True):
        tau = to_quantity(radprops.sigma_t.sum()) * delta_z

    assert np.isclose(tau, tau_ref)


@pytest.mark.parametrize(
    "tau_ref",
    np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * ureg.dimensionless,
)
def test_particle_layer_eval_sigma_t_mono(
    mode_mono,
    tau_ref,
    test_dataset_path,
):
    r"""
    Spectral dependency of extinction is accounted for.

    If :math:`\sigma_t(\lambda)` denotes the extinction coefficient at the
    wavelength :math:`\lambda`, then the optical thickness of a uniform
    particle layer is :math:`\tau(\lambda) = \sigma_t(\lambda) \, \Delta z`
    where :math:`\Delta z` is the layer's thickness.
    It follows that:

    .. math::

       \frac{\tau(\lambda)}{\tau(\lambda_{\mathrm{ref}})} =
       \frac{\sigma(\lambda)}{\sigma(\lambda_{\mathrm{ref}})}

    which is what we assert in this test.
    """
    w_ref = 550 * ureg.nm
    layer = ParticleLayer(
        dataset=test_dataset_path,
        bottom=0.5 * ureg.km,  # arbitrary
        top=3.0 * ureg.km,  # arbitrary
        distribution={"type": "uniform"},
        n_layers=1,
        w_ref=w_ref,
        tau_ref=tau_ref,
    )

    # layer optical thickness @ current wavelength
    wavelengths = np.linspace(500.0, 1500.0, 101) * ureg.nm
    tau = layer.eval_sigma_t_mono(wavelengths) * layer.height

    # data set extinction @ running and reference wavelength
    ds = converters.load_dataset(test_dataset_path)
    w_units = ureg(ds.w.attrs["units"])
    sigma_t = to_quantity(ds.sigma_t.interp(w=wavelengths.m_as(w_units)))
    sigma_t_ref = to_quantity(ds.sigma_t.interp(w=w_ref.m_as(w_units)))

    # the spectral dependence of the optical thickness and extinction
    # coefficient match, so the below ratios must match
    assert np.allclose(tau / tau_ref, sigma_t / sigma_t_ref)
