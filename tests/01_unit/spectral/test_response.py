import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF


def test_uniform_srf():
    # Default constructor succeeds
    assert UniformSRF() is not None

    # Evaluation works as expected
    srf = UniformSRF(1.0, 400.0, 500.0)
    value = srf.eval([300, 400, 450, 500, 600]).m
    expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(value, expected)


def test_delta_srf():
    # Construct from float
    srf = DeltaSRF(0.55 * ureg.micron)
    np.testing.assert_array_equal(
        srf.wavelengths.m_as("nm"),
        [550.0],
        strict=True,
    )

    # Construct from array
    srf = DeltaSRF([440.0, 550.0, 660.0])
    np.testing.assert_array_equal(
        srf.wavelengths.m_as("nm"),
        [440.0, 550.0, 660.0],
        strict=True,
    )

    # Evaluate with single value
    np.testing.assert_array_equal(
        srf.eval(500.0).m_as("dimensionless"),
        0.0,
        strict=True,
    )

    # Evaluate with multiple values
    np.testing.assert_array_equal(
        srf.eval([500.0]).m_as("dimensionless"),
        [0.0],
        strict=True,
    )

    np.testing.assert_array_equal(
        srf.eval([440.0, 550.0, 660.0]).m_as("dimensionless"),
        [0.0, 0.0, 0.0],
        strict=True,
    )


def test_band_srf():
    # Construct
    srf = BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])

    # Construct without leading and trailing zeros
    with pytest.warns(UserWarning):
        BandSRF(wavelengths=[500, 550, 600], values=[1, 1, 1])

    # Evaluate with single value
    np.testing.assert_array_equal(
        srf.eval(550.0).m_as("dimensionless"), 1.0, strict=True
    )

    # Evaluate with multiple values
    values = srf.eval([400, 500, 525, 550, 575, 600, 700]).m
    expected = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
    np.testing.assert_array_equal(values, expected)

    # Construct from ID
    srf = BandSRF.from_id("sentinel_2a-msi-3")
    np.testing.assert_array_equal(srf.support().m_as("nm"), [537, 584])

    # Integrate
    np.testing.assert_equal(
        BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])
        .integrate(500, 600)
        .m_as("nm"),
        50.0,
    )
    np.testing.assert_equal(
        BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])
        .integrate(500, 550)
        .m_as("nm"),
        25.0,
    )
    np.testing.assert_equal(
        BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])
        .integrate(500, 550)
        .m_as("nm"),
        25.0,
    )
