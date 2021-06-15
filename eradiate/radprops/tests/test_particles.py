import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops.particles import ParticleLayer
from eradiate.radprops.particle_dist import UniformParticleDistribution


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


def test_particle_layer_from_dict() -> None:
    """Assigns parameters to expected values."""
    config = dict(
        bottom=ureg.Quantity(1.2, "km"),
        top=ureg.Quantity(1.8, "km"),
        distribution=UniformParticleDistribution(),
        tau_550=0.3,
        n_layers=1,
        dataset="tests/radprops/rtmom_aeronet_desert.nc",
    )
    layer = ParticleLayer.from_dict(config)
    assert isinstance(layer, ParticleLayer)


def test_particle_layer_altitude_units() -> None:
    """Accept different units for bottom and top altitudes."""
    bottom = ureg.Quantity(1, "km")
    top = ureg.Quantity(2000.0, "m")
    config = dict(bottom=bottom, top=top)
    layer = ParticleLayer.from_dict(config)
    assert layer.bottom == bottom
    assert layer.top == top


def test_particle_layer_invalid_bottom_top() -> None:
    """Raises when 'bottom' and 'top' are invalid."""
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    config = {
        "bottom": top,
        "top": bottom,
    }
    with pytest.raises(ValueError):
        ParticleLayer.from_dict(config)


def test_particle_layer_invalid_tau_550() -> None:
    """Raises when 'tau_550' is invalid."""
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    config = {
        "bottom": bottom,
        "top": top,
        "tau_550": -0.1,
    }
    with pytest.raises(ValueError):
        ParticleLayer.from_dict(config)


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
