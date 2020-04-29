import numpy as np
import pytest

from eradiate.scenes.builder.base import Float, Int
from eradiate.scenes.builder.spectra import Spectrum


def test_spectrum_init():
    # Init with float
    s = Spectrum(1.0)
    assert np.allclose(s.value, [1.0])
    assert s.to_xml().strip() == """<spectrum value="1.0"/>"""

    # Init as named parameter, with vector
    s = Spectrum(name="irradiance", value=[0.0, 0.5, 1.0])
    assert np.allclose(s.value, [0.0, 0.5, 1.0])
    assert s.to_xml().strip() == """<spectrum name="irradiance" value="0.0, 0.5, 1.0"/>"""


def test_spectrum_add():
    # Add two spectra
    s1 = Spectrum([1, 1, 1])
    s2 = Spectrum([2, 2, 2])
    s = s1 + s2
    assert np.allclose(s.value, [3.0, 3.0, 3.0])

    # Add spectrum and other type
    s = Spectrum([1, 1, 1])

    for x in [2, 2.]:
        for result in [x + s, s + x]:
            assert isinstance(result, Spectrum)
            assert np.allclose(result.value, [3.0, 3.0, 3.0])

    for x in [Int(2), Float(2.)]:
        with pytest.raises(TypeError):
            result = x + s
        with pytest.raises(TypeError):
            s + x


def test_spectrum_mul():
    # Multiply two spectra
    s1 = Spectrum([3, 3, 3])
    s2 = Spectrum([2, 2, 2])
    s = s1 * s2
    assert np.allclose(s.value, [6.0, 6.0, 6.0])

    # Multiply spectrum and other type
    s = Spectrum([3, 3, 3])

    for x in [2, 2.]:
        for result in [x * s, s * x]:
            assert isinstance(result, Spectrum)
            assert np.allclose(result.value, [6.0, 6.0, 6.0])

    for x in [Int(2), Float(2.)]:
        with pytest.raises(TypeError):
            result = x * s
        with pytest.raises(TypeError):
            s * x

    # Multiply spectrum and other type
    s = Spectrum([3, 3, 3])

    for x in [2, 2.]:
        for result in [x * s, s * x]:
            assert isinstance(result, Spectrum)
            assert np.allclose(result.value, [6.0, 6.0, 6.0])

    for x in [Int(2), Float(2.)]:
        with pytest.raises(TypeError):
            result = x * s
        with pytest.raises(TypeError):
            s * x


def test_spectrum_sub():
    # Negate spectrum
    s = Spectrum([1, 2, 3, 4])
    assert np.allclose(-s.value, [-1, -2, -3, -4])

    # Subtract two spectra
    s1 = Spectrum([3, 3, 3])
    s2 = Spectrum([2, 2, 2])
    s = s1 - s2
    assert np.allclose(s.value, [1, 1, 1])

    # Substract spectrum and other type
    s = Spectrum([3, 3, 3])

    for x in [2, 2.]:
        result = x - s
        assert isinstance(result, Spectrum)
        assert np.allclose(result.value, [-1, -1, -1])

        result = s - x
        assert isinstance(result, Spectrum)
        assert np.allclose(result.value, [1, 1, 1])

    for x in [Int(2), Float(2.)]:
        with pytest.raises(TypeError):
            result = x - s
        with pytest.raises(TypeError):
            s - x


def test_spectrum_div():
    # Invert spectrum
    s = Spectrum([1, 2, 3, 4])
    assert np.allclose(1 / s.value, [1, 0.5, 1. / 3., 0.25])

    # Subtract two spectra
    s1 = Spectrum([3, 3, 3])
    s2 = Spectrum([2, 2, 2])
    s = s1 / s2
    assert np.allclose(s.value, [1.5, 1.5, 1.5])

    # Divide spectrum and other type
    s = Spectrum([3, 3, 3])

    for x in [2, 2.]:
        result = x / s
        assert isinstance(result, Spectrum)
        assert np.allclose(result.value, [2. / 3., 2. / 3., 2. / 3.])

        result = s / x
        assert isinstance(result, Spectrum)
        assert np.allclose(result.value, [1.5, 1.5, 1.5])

    for x in [Int(2), Float(2.)]:
        with pytest.raises(TypeError):
            result = x / s
        with pytest.raises(TypeError):
            s / x
