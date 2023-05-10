import numpy as np
import pytest

from eradiate.scenes.atmosphere import (
    ArrayParticleDistribution,
    ExponentialParticleDistribution,
    GaussianParticleDistribution,
    UniformParticleDistribution,
)


def test_particle_dist_uniform_construct():
    # Constructing without argument is possible
    UniformParticleDistribution()

    # 'bounds' must be sorted
    with pytest.raises(ValueError):
        UniformParticleDistribution([1.0, 0.0])

    # 'bounds' must be a 2-array
    with pytest.raises(ValueError):
        UniformParticleDistribution([0.0, 0.5, 1.0])


def test_particle_dist_uniform():
    dist = UniformParticleDistribution(bounds=[0.0, 0.5])
    x = np.linspace(-0.5, 1.0, 7)
    expected = np.array([0, 0, 2, 2, 2, 0, 0])
    assert np.allclose(dist(x), expected)


@pytest.mark.parametrize(
    "rate, scale, raises",
    [
        (None, None, False),
        (5.0, None, False),
        (None, 0.2, False),
        (1.0, 1.0, True),
    ],
)
def test_particle_dist_exponential(rate, scale, raises):
    if raises:
        with pytest.raises(ValueError):
            ExponentialParticleDistribution(rate=rate, scale=scale)
    else:
        dist = ExponentialParticleDistribution(rate=rate, scale=scale)
        x = np.linspace(0, 1, 11)
        expected = 5.0 * np.exp(-5.0 * x) / (1.0 - np.exp(-5.0))
        assert np.allclose(dist(x), expected)


def test_particle_dist_gaussian():
    dist = GaussianParticleDistribution(mean=0.0, std=1.0)
    x = np.linspace(0, 1, 11)
    expected = np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)
    assert np.allclose(dist(x), expected)


def test_particle_dist_array_construct():
    # 'values' field is required
    with pytest.raises(TypeError):
        ArrayParticleDistribution()

    # Omitting the 'coords' field sets a default, regularly spaced grid
    dist = ArrayParticleDistribution(values=[0.1, 0.2, 0.3, 0.2, 0.1])
    assert np.allclose([0.1, 0.3, 0.5, 0.7, 0.9], dist.coords)

    # Specifying the 'coords' field requires both 'values' and 'coords' to be
    # of the same length
    with pytest.raises(ValueError):
        ArrayParticleDistribution(
            values=[0.1, 0.2, 0.3, 0.2, 0.1], coords=np.linspace(0, 1, 6)
        )


@pytest.mark.parametrize(
    "method, expected",
    [
        ("nearest", [0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]),
        ("nearest-up", [0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
        ("linear", [0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.25, 0.2, 0.15, 0.1, 0.1]),
        ("zero", [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.1]),
        ("slinear", [0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.25, 0.2, 0.15, 0.1, 0.1]),
        ("quadratic", [0.1, 0.1, 0.145, 0.2, 0.265, 0.3, 0.265, 0.2, 0.145, 0.1, 0.1]),
        (
            "cubic",
            [0.1, 0.1, 0.13125, 0.2, 0.26875, 0.3, 0.26875, 0.2, 0.13125, 0.1, 0.1],
        ),
        ("previous", [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.1]),
        ("next", [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
    ],
    ids=[
        "nearest",
        "nearest-up",
        "linear",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ],
)
def test_particle_dist_array_call(method, expected):
    dist = ArrayParticleDistribution(
        values=[0.1, 0.2, 0.3, 0.2, 0.1],
        method=method,
        extrapolate="nearest",
    )
    x = np.linspace(0, 1, 11)
    result = dist(x)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "extrapolate, expected",
    [
        ("zero", [0, 0]),
        ("nearest", [0.1, 0.1]),
        ("method", [0.05, 0.05]),
        ("nan", None),
    ],
    ids=["zero", "nearest", "method", "nan"],
)
def test_particle_dist_array_extrapolate(extrapolate, expected):
    dist = ArrayParticleDistribution(
        values=[0.1, 0.2, 0.3, 0.2, 0.1],
        method="linear",
        extrapolate=extrapolate,
    )
    result = dist([0.0, 1.0])

    if extrapolate != "nan":
        assert np.allclose(result, expected)
    else:
        assert np.isnan(result).all()
