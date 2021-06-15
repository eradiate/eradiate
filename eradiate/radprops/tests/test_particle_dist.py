import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.particle_dist import (
    ArrayParticleDistribution,
    ExponentialParticleDistribution,
    GaussianParticleDistribution,
    UniformParticleDistribution,
)

# ------------------------------------------------------------------------------
#                   Uniform vertical distribution
# ------------------------------------------------------------------------------


@pytest.fixture
def test_altitudes():
    bottom = ureg.Quantity(0000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    return np.linspace(bottom, top)


def test_uniform():
    """Constructor returns 'Uniform' object."""
    dist = UniformParticleDistribution()
    assert isinstance(dist, UniformParticleDistribution)


def test_uniform_from_dict():
    "Class method 'from_dict' returns a 'Uniform' object."
    dist = UniformParticleDistribution.from_dict({})
    assert isinstance(dist, UniformParticleDistribution)


def test_uniform_fractions_all_equal(test_altitudes):
    """Particle fraction values are all equal."""
    dist = UniformParticleDistribution()
    fractions = dist.eval_fraction(test_altitudes)
    assert (fractions == fractions[0]).all()


def test_uniform_fractions_values_sum_to_one(test_altitudes):
    """Particle fraction values sum to one."""
    dist = UniformParticleDistribution()
    fractions = dist.eval_fraction(test_altitudes)
    assert np.isclose(np.sum(fractions), 1.0, rtol=1e-6)


# ------------------------------------------------------------------------------
#                   Gaussian vertical distribution
# ------------------------------------------------------------------------------


def test_gaussian():
    "Assign 'mean' and 'std' attributes."
    mean = ureg.Quantity(1000.0, "m")
    std = ureg.Quantity(200.0, "m")
    dist = GaussianParticleDistribution(mean=mean, std=std)
    assert dist.mean == mean
    assert dist.std == std


def test_gaussian_fractions_sum_to_one(test_altitudes):
    """Particle fractions sum to one."""
    mean = ureg.Quantity(1000.0, "m")
    std = ureg.Quantity(200.0, "m")
    dist = GaussianParticleDistribution(mean=mean, std=std)
    f = dist.eval_fraction(test_altitudes)
    assert np.isclose(np.sum(f), 1.0, rtol=1e-6)


# ------------------------------------------------------------------------------
#                   Exponential vertical distribution
# ------------------------------------------------------------------------------


def test_exponential():
    """Bottom and top altitudes are assigned."""
    rate = ureg.Quantity(1.0, "km^-1")
    dist = ExponentialParticleDistribution(rate=rate)
    assert dist.rate == rate


def test_exponential_fractions_sum_to_one(test_altitudes):
    """Particle fractions sum to one."""
    rate = ureg.Quantity(1.0, "km^-1")
    dist = ExponentialParticleDistribution(rate=rate)
    f = dist.eval_fraction(test_altitudes)
    assert np.isclose(np.sum(f), 1.0, rtol=1e-6)


# ------------------------------------------------------------------------------
#                   Array vertical distribution
# ------------------------------------------------------------------------------


def test_array():
    """Bottom and top altitudes are assigned."""
    values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    dist = ArrayParticleDistribution(values=values)
    assert np.allclose(dist.values, values)


def test_array_fractions_sum_to_one(test_altitudes):
    """Particle fractions sum to one."""
    values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    dist = ArrayParticleDistribution(values=values)
    f = dist.eval_fraction(test_altitudes)
    assert np.isclose(np.sum(f), 1.0, rtol=1e-6)


def test_array_fill_value_is_zero():
    """Missing values in 'data_array' are replaced by zero."""
    bottom = ureg.Quantity(200.0, "m")
    top = ureg.Quantity(800.0, "m")
    z_values = np.linspace(bottom.magnitude + 100, top.magnitude)
    da = xr.DataArray(
        data=np.random.random(len(z_values)),
        coords={
            "z": ("z", z_values, {"units": "m"}),
        },
        dims=["z"],
    )
    dist = ArrayParticleDistribution(data_array=da)
    f = dist.eval_fraction(np.linspace(bottom, top))
    assert np.isclose(f[0], 0.0, rtol=1e-6)


def test_array_invalid_values():
    """Raises when negative values are passed."""
    values = np.array([1.0, -2.0, 3.0, 2.0, -1.0])
    with pytest.raises(ValueError):
        ArrayParticleDistribution(values=values)


def test_array_invalid_data_array():
    """Raises when negative values are passed to 'data_array'."""
    da = xr.DataArray(
        data=np.array([1.0, -2.0, 3.0, 2.0, -0.01, 0.0, 5.0]),
        coords={
            "z": ("z", np.linspace(200, 800, 7), {"units": "m"}),
        },
        dims=["z"],
    )
    with pytest.raises(ValueError):
        ArrayParticleDistribution(data_array=da)
