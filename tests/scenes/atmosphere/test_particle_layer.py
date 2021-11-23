import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.atmosphere._particle_dist import UniformParticleDistribution
from eradiate.scenes.atmosphere._particle_layer import ParticleLayer
from eradiate.units import to_quantity


def test_particle_layer_construct_basic(mode_mono):
    """Construction succeeds with basic parameters."""
    assert ParticleLayer(n_layers=9)


def test_particle_layer_scale(mode_mono):
    ctx = KernelDictContext()
    d = ParticleLayer(scale=2.0).kernel_dict(ctx)
    assert d["medium_atmosphere"]["scale"] == 2.0
    assert d.load()


def test_particle_layer_construct_attrs():
    """Assigns parameters to expected values."""
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    tau_550 = ureg.Quantity(0.3, "dimensionless")
    layer = ParticleLayer(
        bottom=bottom,
        top=top,
        distribution=UniformParticleDistribution(),
        tau_550=tau_550,
        n_layers=1,
        dataset="tests/radprops/rtmom_aeronet_desert.nc",
    )
    assert layer.bottom == bottom
    assert layer.top == top
    assert isinstance(layer.distribution, UniformParticleDistribution)
    assert layer.tau_550 == tau_550
    assert layer.n_layers == 1
    assert layer.dataset == path_resolver.resolve(
        "tests/radprops/rtmom_aeronet_desert.nc"
    )


def test_particle_layer_altitude_units():
    """Accept different units for bottom and top altitudes."""
    assert ParticleLayer(bottom=ureg.Quantity(1, "km"), top=ureg.Quantity(2000.0, "m"))


def test_particle_layer_invalid_bottom_top():
    """Raises when 'bottom' is larger that 'top'."""
    with pytest.raises(ValueError):
        ParticleLayer(top=ureg.Quantity(1.2, "km"), bottom=ureg.Quantity(1.8, "km"))


def test_particle_layer_invalid_tau_550():
    """Raises when 'tau_550' is invalid."""
    with pytest.raises(ValueError):
        ParticleLayer(
            bottom=ureg.Quantity(1.2, "km"), top=ureg.Quantity(1.8, "km"), tau_550=-0.1
        )


def test_particle_layer_kernel_phase(modes_all_single):
    atmosphere = ParticleLayer(n_layers=9)

    # Phase function kernel dictionary can be generated
    ctx = KernelDictContext()
    kernel_phase = atmosphere.kernel_phase(ctx)

    # Dictionary key is set to appropriate value
    assert set(kernel_phase.data.keys()) == {f"phase_{atmosphere.id}"}


def test_particle_layer_kernel_dict(modes_all_single):
    particle_layer = ParticleLayer(n_layers=9)

    # Produced kernel dictionary can be loaded
    ctx = KernelDictContext()
    assert particle_layer.kernel_dict(ctx).load()


@pytest.fixture
def test_dataset():
    return path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc")


def test_particle_layer_eval_radprops(mode_mono, test_dataset):
    """Method 'eval_radprops' returns dataset with expected datavars and coords."""
    layer = ParticleLayer(dataset=test_dataset)
    spectral_ctx = SpectralContext.new()
    ds = layer.eval_radprops(spectral_ctx)
    expected_data_vars = ["sigma_t", "albedo"]
    expected_coords = ["z_layer"]
    assert all([coord in ds.coords for coord in expected_coords]) and all(
        [var in ds.data_vars for var in expected_data_vars]
    )


@pytest.mark.parametrize("tau_550", [0.1, 0.5, 1.0, 5.0])
def test_particle_layer_eval_sigma_t(mode_mono, tau_550, test_dataset):
    r"""Spectral dependance of extinction is accounted for.

    If :math:`\sigma_t(\lambda)` denotes the extinction coefficient at the
    wavelength :math:`\lambda`, then the optical thickness of a uniform
    particle layer is :math:`\tau(\lambda) = \sigma_t(\lambda) \, \Delta z`
    where :math:`\Delta z` is the layer's thickness.
    It follows that:

    .. math::

       \frac{\tau(\lambda)}{\tau(550\, \mathrm{nm})} =
       \frac{\sigma(\lambda)}{\sigma(550\, \mathrm{nm})}

    which is what we assert in this test.
    """
    wavelengths = np.linspace(500.0, 1500.0, 11) * ureg.nm
    tau_550 = tau_550 * ureg.dimensionless

    # tau_550 = 1.0 * ureg.dimensionless
    layer = ParticleLayer(
        dataset=test_dataset,
        bottom=0.0 * ureg.km,
        top=1.0 * ureg.km,
        distribution={"type": "uniform"},
        n_layers=1,
        tau_550=tau_550,
    )

    # layer optical thickness @ current wavelengths
    tau = np.empty_like(wavelengths) * ureg.dimensionless

    for i, w in enumerate(wavelengths):
        spectral_ctx = SpectralContext.new(wavelength=w)
        tau[i] = layer.eval_sigma_t(spectral_ctx) * layer.height

    # data set extinction @ running wavelength and 550 nm
    with xr.open_dataset(test_dataset) as ds:
        w_units = ureg(ds.w.attrs["units"])
        sigma_t = to_quantity(ds.sigma_t.interp(w=wavelengths.m_as(w_units)))
        sigma_t_550 = to_quantity(ds.sigma_t.interp(w=(550.0 * ureg.nm).m_as(w_units)))

    # the spectral dependence of the optical thickness and extinction coefficient
    # match, so the below ratios must match
    assert np.allclose(tau / tau_550, sigma_t / sigma_t_550)
