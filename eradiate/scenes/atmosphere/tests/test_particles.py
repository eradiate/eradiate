import pathlib

import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.atmosphere._particle_dist import UniformParticleDistribution
from eradiate.scenes.atmosphere._particles import ParticleLayer
from eradiate.scenes.core import KernelDict


def test_particle_load(mode_mono):
    """Produces a kernel dictionary that can be loaded by the kernel."""
    ctx = KernelDictContext()
    particle_layer = ParticleLayer(n_layers=9)
    assert KernelDict.from_elements(particle_layer, ctx=ctx).load() is not None


def test_particle_layer():
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


def test_particle_layer_eval_phase(mode_mono, test_dataset):
    """Method 'eval_phase' returns a 'DataArray'."""
    layer = ParticleLayer(dataset=test_dataset)
    spectral_ctx = SpectralContext.new()
    assert isinstance(layer.eval_phase(spectral_ctx=spectral_ctx), xr.DataArray)


def test_particle_layer_eval_phase_order(mode_mono, tmpdir):
    """
    Returns phase function values by increasing order of scattering anle cosine
    values.
    """

    def make_ds(mu, phase):
        return xr.Dataset(
            data_vars={
                "sigma_t": ("w", np.zeros(2)),
                "albedo": ("w", np.zeros(2)),
                "phase": (
                    ["w", "mu", "i", "j"],
                    phase[:, :, np.newaxis, np.newaxis],
                ),
            },
            coords={
                "w": ("w", [240.0, 2800.0], dict(units="nm")),
                "mu": ("mu", mu),
                "i": ("i", [0]),
                "j": ("j", [0]),
            },
        )

    ds_mu_increasing = make_ds(
        mu=np.linspace(-1, 1, 3), phase=np.array([np.arange(1, 4), np.arange(1, 4)])
    )
    path_mu_increasing = pathlib.Path(tmpdir, "ds_mu_increasing.nc")
    ds_mu_increasing.to_netcdf(path_mu_increasing)

    ds_mu_decreasing = make_ds(
        mu=np.linspace(1, -1, 3),
        phase=np.array([np.arange(3, 0, -1), np.arange(3, 0, -1)]),
    )
    path_mu_decreasing = pathlib.Path(tmpdir, "ds_mu_decreasing.nc")
    ds_mu_decreasing.to_netcdf(path_mu_decreasing)

    spectral_ctx = SpectralContext.new()
    layer_mu_increasing = ParticleLayer(dataset=path_mu_increasing)
    phase_mu_increasing = layer_mu_increasing.eval_phase(spectral_ctx)

    layer_mu_decreasing = ParticleLayer(dataset=path_mu_decreasing)
    phase_mu_decreasing = layer_mu_decreasing.eval_phase(spectral_ctx)

    assert np.all(phase_mu_increasing.values == phase_mu_decreasing.values)
