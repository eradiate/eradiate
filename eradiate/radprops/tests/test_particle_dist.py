import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.particle_dist import (
    Array,
    Exponential,
    Gaussian,
    Uniform,
)

# ------------------------------------------------------------------------------
#                   Uniform vertical distribution
# ------------------------------------------------------------------------------


def test_uniform():
    "Assign bottom and top attributes."
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Uniform(bottom=bottom, top=top)
    assert dist.bottom == bottom
    assert dist.top == top


def test_uniform_fractions_all_equal():
    """Particle fraction values are all equal."""
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Uniform(bottom=bottom, top=top)
    fractions = dist.eval_fraction(np.linspace(bottom, top))
    assert (fractions == fractions[0]).all()


def test_uniform_fractions_values_sum_to_one():
    """Particle fraction values sum to one."""
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Uniform(bottom=bottom, top=top)
    fractions = dist.eval_fraction(np.linspace(bottom, top))
    assert np.isclose(np.sum(fractions), 1.0, rtol=1e-6)


def test_uniform_invalid_altitude_1():
    "Invalid altitude value passed to 'fractions' raises."
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Uniform(bottom=bottom, top=top)
    with pytest.raises(ValueError):
        dist.eval_fraction(np.linspace(bottom - ureg.Quantity(1.0, "m"), top))


def test_uniform_invalid_altitude_2():
    "Invalid altitude value passed to 'fractions' raises."
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Uniform(bottom=bottom, top=top)
    with pytest.raises(ValueError):
        dist.eval_fraction(np.linspace(bottom, top + ureg.Quantity(1.0, "m")))


def test_uniform_invalid_bottom_top():
    "Invalid values passed to 'bottom' and 'top' raise."
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    with pytest.raises(ValueError):
        Uniform(bottom=top, top=bottom)


def test_uniform_invalid_bottom_top_2():
    "Invalid values passed to 'bottom' and 'top' raise."
    bottom = ureg.Quantity(1000.0, "m")
    with pytest.raises(ValueError):
        Uniform(bottom=bottom, top=bottom)


# ------------------------------------------------------------------------------
#                   Gaussian vertical distribution
# ------------------------------------------------------------------------------


def test_gaussian():
    "Assign bottom and top attributes."
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Gaussian(bottom=bottom, top=top)
    assert dist.bottom == bottom
    assert dist.top == top


def test_gaussian_default_mean_and_std():
    """Default 'mean' and 'std' parameters computed according to docstrings."""
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Gaussian(bottom=bottom, top=top)
    assert dist.mean == (bottom + top) / 2.0
    assert dist.std == (top - bottom) / 6.0


def test_gaussian_fractions_sum_to_one():
    """Particle fractions sum to one."""
    bottom = ureg.Quantity(1000.0, "m")
    top = ureg.Quantity(2000.0, "m")
    dist = Gaussian(bottom=bottom, top=top)
    f = dist.eval_fraction(np.linspace(bottom, top))
    assert np.isclose(np.sum(f), 1.0, rtol=1e-6)


# ------------------------------------------------------------------------------
#                   Exponential vertical distribution
# ------------------------------------------------------------------------------


def test_exponential():
    """Bottom and top altitudes are assigned."""
    bottom = ureg.Quantity(0.0, "m")
    top = ureg.Quantity(500.0, "m")
    dist = Exponential(bottom=0.0, top=500.0)
    assert dist.bottom == bottom
    assert dist.top == top


def test_exponential_default_rate():
    """Default 'rate' parameters computed according to docs."""
    bottom = ureg.Quantity(0.0, "m")
    top = ureg.Quantity(500.0, "m")
    dist = Exponential(bottom=0.0, top=500.0)
    assert dist.rate == 1.0 / (top - bottom)


def test_exponential_fractions_sum_to_one():
    """Particle fractions sum to one."""
    bottom = ureg.Quantity(0.0, "m")
    top = ureg.Quantity(500.0, "m")
    dist = Exponential(bottom=0.0, top=500.0)
    f = dist.eval_fraction(np.linspace(bottom, top))
    assert np.isclose(np.sum(f), 1.0, rtol=1e-6)


# ------------------------------------------------------------------------------
#                   Array vertical distribution
# ------------------------------------------------------------------------------


def test_array():
    """Bottom and top altitudes are assigned."""
    bottom = ureg.Quantity(200.0, "m")
    top = ureg.Quantity(820.0, "m")
    dist = Array(bottom=bottom, top=top, values=[1.0, 2.0, 3.0, 2.0, 1.0])
    assert dist.bottom == bottom
    assert dist.top == top


def test_array_fractions_sum_to_one():
    """Particle fractions sum to one."""
    bottom = ureg.Quantity(200.0, "m")
    top = ureg.Quantity(820.0, "m")
    dist = Array(bottom=bottom, top=top, values=[1.0, 2.0, 3.0, 2.0, 1.0])
    f = dist.eval_fraction(np.linspace(bottom, top))
    assert np.isclose(np.sum(f), 1.0, rtol=1e-6)


def test_array_fill_value_is_zero():
    """Missing values in 'data_array' are replaced by zero."""
    bottom = ureg.Quantity(200.0, "m")
    top = ureg.Quantity(820.0, "m")
    z_values = np.linspace(bottom.magnitude + 1, top.magnitude)
    da = xr.DataArray(
        data=np.random.random(len(z_values)),
        coords={
            "z": ("z", z_values, {"units": "m"}),
        },
        dims=["z"],
    )
    dist = Array(bottom=bottom, top=top, data_array=da)
    f = dist.eval_fraction(np.linspace(bottom, top))
    assert np.isclose(f[0], 0.0, rtol=1e-6)


def test_array_invalid():
    """Invalid 'data_array' raises."""
    bottom = ureg.Quantity(200.0, "m")
    top = ureg.Quantity(820.0, "m")
    z_values = np.linspace(bottom.magnitude - 100, top.magnitude)
    da = xr.DataArray(
        data=np.random.random(len(z_values)),
        coords={
            "z": ("z", z_values, {"units": "m"}),
        },
        dims=["z"],
    )
    with pytest.raises(ValueError):
        Array(bottom=bottom, top=top, data_array=da)
