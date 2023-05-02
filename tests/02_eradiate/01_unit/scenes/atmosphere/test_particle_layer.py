import numpy as np
import pytest
import xarray as xr

from eradiate import KernelContext, data
from eradiate import unit_registry as ureg
from eradiate.radprops import ZGrid
from eradiate.scenes.atmosphere import ParticleLayer, UniformParticleDistribution
from eradiate.scenes.core import traverse
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element
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


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_particle_layer_kernel_dict(mode_mono, geometry):
    """
    Kernel dictionary checks
    """
    layer = ParticleLayer(geometry=geometry)
    check_scene_element(layer)


@pytest.mark.parametrize("w", [280.0, 550.0, 1600.0, 2400.0] * ureg.nm)
def test_particle_layer_eval_mono_absorbing_only(mode_mono, absorbing_only, w):
    """eval methods return expected values for an absorbing-only layer."""
    bottom = 0.0 * ureg.km
    top = 1.0 * ureg.km
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        dataset=absorbing_only,
        bottom=bottom,
        top=top,
    )
    si = SpectralIndex.new(w=w)
    assert np.allclose(layer.eval_sigma_s(si), 0.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(si), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(si).m, 0.0)

    ctx = KernelContext(si=si)
    assert layer.eval_mfp(ctx).magnitude == np.inf


@pytest.mark.parametrize("w", [280.0, 550.0, 1600.0, 2400.0] * ureg.nm)
def test_particle_layer_eval_mono_scattering_only(mode_mono, scattering_only, w):
    """eval methods return expected values for a scattering-only layer."""
    bottom = 0.0 * ureg.km
    top = 1.0 * ureg.km
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        dataset=scattering_only,
        bottom=bottom,
        top=top,
    )
    si = SpectralIndex.new(w=w)
    assert np.allclose(layer.eval_sigma_s(si), 0.2 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(si), 0.0 / ureg.km)
    assert np.allclose(layer.eval_albedo(si).m, 1.0)

    ctx = KernelContext(si=si)
    assert np.isclose(layer.eval_mfp(ctx), 5.0 * ureg.km)


@pytest.mark.parametrize("w", [280.0, 550.0, 1600.0, 2400.0] * ureg.nm)
def test_particle_layer_eval_mono(mode_mono, test_particles_dataset, w):
    """
    eval_* methods return expected values for a scattering and absorbing layer.
    """
    bottom = 0.0 * ureg.km
    top = 1.0 * ureg.km
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        dataset=test_particles_dataset,
        tau_ref=1.0,
        bottom=bottom,
        top=top,
    )
    si = SpectralIndex.new(w=w)
    assert np.allclose(layer.eval_sigma_t(si), 1.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_s(si), 0.8 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(si), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(si).m, 0.8)

    ctx = KernelContext(si=si)
    assert np.isclose(layer.eval_mfp(ctx), 1.25 * ureg.km)


@pytest.mark.parametrize("w", [280.0, 550.0, 1600.0, 2400.0] * ureg.nm)
def test_particle_layer_eval_ckd_absorbing_only(mode_ckd, absorbing_only, w):
    """eval methods return expected values for an absorbing-only layer."""
    bottom = 0.0 * ureg.km
    top = 1.0 * ureg.km
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        dataset=absorbing_only,
        bottom=bottom,
        top=top,
    )
    si = SpectralIndex.new(w=w)
    assert np.allclose(layer.eval_sigma_s(si), 0.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(si), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(si).m, 0.0)

    ctx = KernelContext(si=si)
    assert layer.eval_mfp(ctx).magnitude > 0.0


@pytest.mark.parametrize("w", [280.0, 550.0, 1600.0, 2400.0] * ureg.nm)
def test_particle_layer_eval_ckd_scattering_only(mode_ckd, scattering_only, w):
    """eval methods return expected values for a scattering-only layer."""
    bottom = 0.0 * ureg.km
    top = 1.0 * ureg.km
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        dataset=scattering_only,
        bottom=bottom,
        top=top,
    )
    si = SpectralIndex.new(w=w)
    assert np.allclose(layer.eval_sigma_s(si), 0.2 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(si).m, 0.0)
    assert np.allclose(layer.eval_albedo(si).m, 1.0)

    ctx = KernelContext(si=si)
    assert layer.eval_mfp(ctx).magnitude > 0.0


@pytest.mark.parametrize("w", [280.0, 550.0, 1600.0, 2400.0] * ureg.nm)
def test_particle_layer_eval_ckd(mode_ckd, test_particles_dataset, w):
    """eval methods return expected values for a scattering-only layer."""
    bottom = 0.0 * ureg.km
    top = 1.0 * ureg.km
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        dataset=test_particles_dataset,
        tau_ref=1.0,
        bottom=bottom,
        top=top,
    )
    si = SpectralIndex.new(w=w)
    assert np.allclose(layer.eval_sigma_t(si), 1.0 / ureg.km)
    assert np.allclose(layer.eval_sigma_s(si), 0.8 / ureg.km)
    assert np.allclose(layer.eval_sigma_a(si), 0.2 / ureg.km)
    assert np.allclose(layer.eval_albedo(si).m, 0.8)

    ctx = KernelContext(si=si)
    assert layer.eval_mfp(ctx) == 1.25 * ureg.km


def test_particle_layer_construct_basic():
    """Construction succeeds with basic parameters."""
    assert ParticleLayer()


def test_particle_layer_scale(modes_all_single):
    """Scale parameter propagates to kernel dict and latter can be loaded."""
    particle_layer = ParticleLayer(id="atmosphere", scale=2.0)
    template, _ = traverse(particle_layer)
    assert template["medium_atmosphere.scale"] == 2.0


def test_particle_layer_construct_attrs(test_dataset_path):
    """Assigns parameters to expected values."""
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    tau_ref = ureg.Quantity(0.3, "dimensionless")
    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "ground_altitude": bottom,
            "toa_altitude": top,
        },
        bottom=bottom,
        top=top,
        distribution=UniformParticleDistribution(),
        tau_ref=tau_ref,
        dataset=test_dataset_path,
    )
    assert layer.bottom_altitude == bottom
    assert layer.top_altitude == top
    assert isinstance(layer.distribution, UniformParticleDistribution)
    assert layer.tau_ref == tau_ref
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


def test_particle_layer_eval_radprops_format(
    modes_all_single,
    test_dataset_path,
):
    """
    Method 'eval_radprops' returns dataset with expected datavars and coords.
    """
    layer = ParticleLayer(dataset=test_dataset_path)
    si = SpectralIndex.new()
    ds = layer.eval_radprops(si)
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
        tau_ref=tau_ref,
    )

    # compute optical thickness at reference wavelength from layer's radprops
    # and check it matches the input tau_ref
    si = SpectralIndex.new(w=layer.w_ref)
    radprops = layer.eval_radprops(si)
    delta_z = layer.geometry.zgrid.layer_height

    with xr.set_options(keep_attrs=True):
        tau = to_quantity(radprops.sigma_t.sum()) * delta_z

    assert np.isclose(tau, tau_ref)


@pytest.mark.parametrize("distribution", ["uniform", "gaussian", "exponential"])
@pytest.mark.parametrize(
    "tau_ref",
    np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * ureg.dimensionless,
)
def test_particle_layer_eval_sigma_t_impl(
    mode_mono, tau_ref, distribution, test_dataset_path
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
    bottom = 1.0 * ureg.km  # arbitrary
    top = 4.0 * ureg.km  # arbitrary
    n_wavelengths = 3
    n_layers = 10
    wavelengths = np.linspace(500.0, 1500.0, n_wavelengths) * ureg.nm
    zgrid = ZGrid(np.linspace(0, 5, n_layers + 1) * ureg.km)

    layer = ParticleLayer(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": zgrid.levels[-1],
            "zgrid": zgrid,
        },
        dataset=test_dataset_path,
        bottom=bottom,
        top=top,
        distribution=distribution,
        w_ref=w_ref,
        tau_ref=tau_ref,
    )

    # layer optical thickness @ current wavelength
    sigma_t = layer._eval_sigma_t_impl(wavelengths, layer.geometry.zgrid)
    assert sigma_t.units.is_compatible_with(ureg("m**-1"))
    assert sigma_t.shape == (n_wavelengths, n_layers)
    tau = np.sum(
        sigma_t * layer.geometry.zgrid.layer_height,
        axis=1,
    )  # Integrate sigma_t * dz vs space coordinate using rectangle method

    # data set extinction @ running and reference wavelength
    ds = data.load_dataset(test_dataset_path)
    w_units = ureg(ds.w.attrs["units"])
    sigma_t = to_quantity(ds.sigma_t.interp(w=wavelengths.m_as(w_units)))
    sigma_t_ref = to_quantity(ds.sigma_t.interp(w=w_ref.m_as(w_units)))

    # the spectral dependence of the optical thickness and extinction
    # coefficient match, so the below ratios must match
    result = (tau / tau_ref).m_as(ureg.dimensionless)
    expected = (sigma_t / sigma_t_ref).m_as(ureg.dimensionless)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "has_absorption, has_scattering, expected",
    [
        (
            True,
            True,
            {
                "albedo": 0.97331806,
                "sigma_t": 1.0,
                "sigma_a": 0.02668194,
                "sigma_s": 0.97331806,
            },
        ),
        (
            False,
            True,
            {
                "albedo": 1.0,
                "sigma_t": 0.97331806,
                "sigma_a": 0.0,
                "sigma_s": 0.97331806,
            },
        ),
        (
            True,
            False,
            {
                "albedo": 0.0,
                "sigma_t": 0.02668194,
                "sigma_a": 0.02668194,
                "sigma_s": 0.0,
            },
        ),
        (False, False, "raise"),
    ],
)
def test_particle_layer_switches(mode_mono, has_absorption, has_scattering, expected):
    try:
        bottom = 0.0 * ureg.km
        top = 1.0 * ureg.km
        particle_layer = ParticleLayer(
            geometry={
                "type": "plane_parallel",
                "ground_altitude": bottom,
                "toa_altitude": top,
            },
            tau_ref=1.0,
            has_absorption=has_absorption,
            has_scattering=has_scattering,
        )
        zgrid = particle_layer.geometry.zgrid
        w = 550.0 * ureg.nm

        np.testing.assert_allclose(
            particle_layer.eval_albedo_mono(w, zgrid).m_as(ureg.dimensionless),
            expected["albedo"],
        )
        np.testing.assert_allclose(
            particle_layer.eval_sigma_t_mono(w, zgrid).m_as("km^-1"),
            expected["sigma_t"],
        )
        np.testing.assert_allclose(
            particle_layer.eval_sigma_a_mono(w, zgrid).m_as("km^-1"),
            expected["sigma_a"],
        )
        np.testing.assert_allclose(
            particle_layer.eval_sigma_s_mono(w, zgrid).m_as("km^-1"),
            expected["sigma_s"],
        )

    except ValueError:
        assert expected == "raise"
