import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.particles import (
    Uniform, Gaussian, Exponential, Array, ParticlesLayer)


def test_uniform_vertical_distribution():
    bottom = ureg.Quantity(1000., "m")
    top = ureg.Quantity(2000., "m")
    dist = Uniform(
        bottom=bottom,
        top=top,
    )
    assert dist.bottom == bottom
    assert dist.top == top
    fractions = dist.fractions(np.linspace(bottom, top))
    assert (fractions == fractions[0]).all()
    assert np.isclose(np.sum(fractions), 1., rtol=1e-6)

    # invalid altitudes raise (in fractions method)
    with pytest.raises(ValueError):
        dist.fractions(np.linspace(bottom - ureg.Quantity(1., "m"), top))

    # invalid bottom and top altitudes raise (in constructor)
    with pytest.raises(ValueError):
        dist = Uniform(bottom=top, top=bottom)


def test_gaussian_vertical_distribution():
    bottom = ureg.Quantity(1000., "m")
    top = ureg.Quantity(2000., "m")
    dist = Gaussian(
        bottom=bottom,
        top=top,
    )
    assert dist.bottom == bottom
    assert dist.top == top
    # default mean and std parameters computed according to docs
    assert dist.mean == (bottom + top) / 2.
    assert dist.std == (top - bottom) / 6.
    fractions = dist.fractions(np.linspace(bottom, top))
    assert np.isclose(np.sum(fractions), 1., rtol=1e-6)


def test_exponential_vertical_distribution():
    bottom = ureg.Quantity(0., "m")
    top = ureg.Quantity(500., "m")
    dist = Exponential(
        bottom=0.,
        top=500.
    )
    assert dist.bottom == bottom
    assert dist.top == top
    # default rate parameters computed according to docs
    assert dist.rate == 1. / (top - bottom)
    fractions = dist.fractions(np.linspace(bottom, top))
    assert np.isclose(np.sum(fractions), 1., rtol=1e-6)


def test_array_vertical_distribution():
    bottom = ureg.Quantity(200., "m")
    top = ureg.Quantity(820., "m")
    dist = Array(
        bottom=bottom,
        top=top,
        values=[1., 2., 3., 2., 1.]
    )
    assert dist.bottom == bottom
    assert dist.top == top
    fractions = dist.fractions(np.linspace(bottom, top))
    assert np.isclose(np.sum(fractions), 1., rtol=1e-6)

    assert dist.bottom == bottom
    assert dist.top == top
    fractions = dist.fractions(np.linspace(bottom, top))
    assert np.isclose(np.sum(fractions), 1., rtol=1e-6)

    # fill value = zero
    z_values = np.linspace(bottom.magnitude + 1, top.magnitude)
    da = xr.DataArray(
        data=np.random.random(len(z_values)),
        coords={
            "z": ("z", z_values, {"units": "m"}),
        },
        dims=["z"],
    )
    dist = Array(
        bottom=bottom,
        top=top,
        data_array=da
    )
    fractions = dist.fractions(np.linspace(bottom, top))
    assert np.isclose(fractions[0], 0., rtol=1e-6)

    # invalid data array
    z_values = np.linspace(bottom.magnitude - 100, top.magnitude)
    da = xr.DataArray(
        data=np.random.random(len(z_values)),
        coords={
            "z": ("z", z_values, {"units": "m"}),
        },
        dims=["z"],
    )
    with pytest.raises(ValueError):
        dist = Array(
            bottom=bottom,
            top=top,
            data_array=da
        )


def test_particles_layer(mode_mono):
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    tau_550 = ureg.Quantity(.3, "dimensionless")
    config = {
        "bottom": bottom,
        "top": top,
        "tau_550": tau_550
    }
    layer = ParticlesLayer.from_dict(config)
    assert layer.bottom == bottom
    assert layer.top == top
    assert layer.tau_550 == tau_550
    assert len(layer.z_layer) == layer.n_layers
    # default vertical distribution is uniform:
    assert (layer.fractions == 1).all()

    # invalid config raises
    config = {
        "bottom": top,
        "top": bottom,
    }
    with pytest.raises(ValueError):
        ParticlesLayer.from_dict(config)

    config = {
        "bottom": bottom,
        "top": top,
        "tau_550": -0.1,
    }
    with pytest.raises(ValueError):
        ParticlesLayer.from_dict(config)

    # number of layers adjust to vertical distribution type
    config = {
        "bottom": bottom,
        "top": top,
        "vert_dist": {
            "type": "exponential",
        }
    }
    layer = ParticlesLayer.from_dict(config)
    assert layer.n_layers == 8
    assert len(layer.z_layer) == layer.n_layers

    config.update({"vert_dist": {"type": "gaussian"}})
    layer = ParticlesLayer.from_dict(config)
    assert layer.n_layers == 16

    config.update({"vert_dist": {"type": "array", "values": [1., 2., 3.]}})
    layer = ParticlesLayer.from_dict(config)
    assert layer.n_layers == 32

    # specify the bottom and top altitudes in different units work fine
    bottom = ureg.Quantity(1, "km")
    top = ureg.Quantity(2000., "m")
    config = dict(bottom=bottom, top=top)
    layer = ParticlesLayer.from_dict(config)
    assert layer.bottom == bottom
    assert layer.top == top

    # to_dataset returns well formatted dataset
    ds = layer.to_dataset()
    assert "sigma_t" in ds.data_vars
    assert "albedo" in ds.data_vars
    assert "z_layer" in ds.coords
    assert "w" in ds.coords
