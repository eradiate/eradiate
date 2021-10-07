import pytest

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.atmosphere._particle_dist import UniformParticleDistribution
from eradiate.scenes.atmosphere._particle_layer import ParticleLayer


def test_particle_layer_construct_basic(mode_mono):
    # Construction succeeds with basic parameters
    assert ParticleLayer(n_layers=9)


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


def test_particle_layer_radprops(mode_mono, test_dataset):
    """Method 'radprops' returns data set with expected data_vars and coords."""
    layer = ParticleLayer(dataset=test_dataset)
    spectral_ctx = SpectralContext.new()
    ds = layer.eval_radprops(spectral_ctx)
    expected_data_vars = ["sigma_t", "albedo"]
    expected_coords = ["z_layer"]
    assert all([coord in ds.coords for coord in expected_coords]) and all(
        [var in ds.data_vars for var in expected_data_vars]
    )
