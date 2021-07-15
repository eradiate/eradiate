import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops.particle_dist import UniformParticleDistribution
from eradiate.radprops.particles import ParticleLayer


def test_particle_layer() -> None:
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
    assert layer.dataset == "tests/radprops/rtmom_aeronet_desert.nc"


def test_particle_layer_altitude_units() -> None:
    """Accept different units for bottom and top altitudes."""
    assert ParticleLayer(bottom=ureg.Quantity(1, "km"), top=ureg.Quantity(2000.0, "m"))


def test_particle_layer_invalid_bottom_top() -> None:
    """Raises when 'bottom' is larger that 'top'."""
    with pytest.raises(ValueError):
        ParticleLayer(top=ureg.Quantity(1.2, "km"), bottom=ureg.Quantity(1.8, "km"))


def test_particle_layer_invalid_tau_550() -> None:
    """Raises when 'tau_550' is invalid."""
    with pytest.raises(ValueError):
        ParticleLayer(ureg.Quantity(1.2, "km"), ureg.Quantity(1.8, "km"), tau_550=-0.1)


@pytest.fixture
def test_dataset():
    return path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc")


def test_particle_layer_radprops(mode_mono, test_dataset) -> None:
    """Method 'radprops' returns data set with expected data_vars and coords."""
    layer = ParticleLayer(dataset=test_dataset)
    spectral_ctx = SpectralContext.new()
    ds = layer.radprops(spectral_ctx)
    expected_data_vars = ["sigma_t", "albedo"]
    expected_coords = ["z_layer", "w"]
    assert all([coord in ds.coords for coord in expected_coords]) and all(
        [var in ds.data_vars for var in expected_data_vars]
    )


def test_particle_layer_eval_phase(test_dataset) -> None:
    """Method 'eval_phase' returns a 'DataArray'."""
    layer = ParticleLayer(dataset=test_dataset)
    spectral_ctx = SpectralContext.new()
    assert isinstance(layer.eval_phase(spectral_ctx=spectral_ctx), xr.DataArray)
