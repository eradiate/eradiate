import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.exceptions import ModeError
from eradiate.spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)


def test_spectral_index_new(modes_all):
    w = 1.0 * ureg.nm
    si = SpectralIndex.new(w=w)
    assert si.w == w


def test_spectral_index_new_mono():
    w = 1.0 * ureg.nm
    mono = SpectralIndex.new(mode="mono", w=w)
    assert isinstance(mono, MonoSpectralIndex)
    assert mono.w == w
    assert mono.formatted_repr == f"{w:g~P}"


def test_spectral_index_new_ckd():
    w = 1.0 * ureg.nm
    g = 0.5
    ckd = SpectralIndex.new(mode="ckd", w=w, g=g)
    assert isinstance(ckd, CKDSpectralIndex)
    assert ckd.w == w
    assert ckd.g == g
    assert ckd.formatted_repr == f"{w:g~P}:{g:g}"


def test_spectral_index_new_invalid():
    """``SpectralIndex.new()`` raises when arguments are invalid."""

    # mode = "invalid"

    with pytest.raises(KeyError):
        mode_invalid = "invalid"
        SpectralIndex.new(mode=mode_invalid)

    # mode = "mono"

    with pytest.raises(TypeError):
        w_invalid = np.array([1.0, 2.0]) * ureg.nm  # w must be scalar quantity
        SpectralIndex.new(mode="mono", w=w_invalid)

    with pytest.raises(ValueError):
        w_invalid = -1.0 * ureg.nm  # w must be positive
        SpectralIndex.new(mode="mono", w=w_invalid)

    # mode = "ckd"

    with pytest.raises(TypeError):
        w_invalid = np.array([1.0, 2.0]) * ureg.nm  # w must be scalar quantity
        SpectralIndex.new(mode="ckd", w=w_invalid)

    with pytest.raises(ValueError):
        w_invalid = -1.0 * ureg.nm  # w must be positive
        SpectralIndex.new(mode="ckd", w=w_invalid)

    with pytest.raises(ValueError):
        g_invalid = 2.0  # g must be between 0 and 1
        SpectralIndex.new(mode="ckd", g=g_invalid)

    with pytest.raises(ValueError):
        g_invalid = -1.0  # g must be between 0 and 1
        SpectralIndex.new(mode="ckd", g=g_invalid)


def test_spectral_index_from_dict_mono():
    w = 1.0 * ureg.nm
    mono = SpectralIndex.from_dict({"mode": "mono", "w": w})
    assert isinstance(mono, MonoSpectralIndex)
    assert mono.w == w
    assert mono.formatted_repr == f"{w:g~P}"


def test_spectral_index_from_dict_ckd():
    w = 1.0 * ureg.nm
    g = 0.5
    ckd = SpectralIndex.from_dict({"mode": "ckd", "w": w, "g": g})
    assert isinstance(ckd, CKDSpectralIndex)
    assert ckd.w == w
    assert ckd.g == g
    assert ckd.formatted_repr == f"{w:g~P}:{g:g}"


def test_spectral_index_convert_mono():
    w = 1.0 * ureg.nm
    value = {"mode": "mono", "w": w}
    converted = SpectralIndex.convert(value)
    assert isinstance(converted, MonoSpectralIndex)
    assert np.allclose(converted.w, w)


def test_spectral_index_convert_ckd():
    w = 1.0 * ureg.nm
    g = 0.5
    value = {"mode": "ckd", "w": w, "g": g}
    converted = SpectralIndex.convert(value)
    assert isinstance(converted, CKDSpectralIndex)
    assert np.allclose(converted.w, w)
    assert np.allclose(converted.g, g)
