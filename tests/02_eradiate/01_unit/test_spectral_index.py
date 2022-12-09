import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.exceptions import ModeError
from eradiate.spectral_index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex


def test_spectral_index_new():
    """Test the ``SpectralIndex.new()`` factory method."""

    # Monochromatic spectral index
    w = 1.0 * ureg.nm
    mono = SpectralIndex.new(mode="mono", w=w)
    assert isinstance(mono, MonoSpectralIndex)
    assert mono.w == w
    assert mono.formatted_repr == f"{w:g~P}"
    
    # CKD spectral index
    w = 1.0 * ureg.nm
    g = 0.5
    ckd = SpectralIndex.new(mode="ckd", w=w, g=g)
    assert isinstance(ckd, CKDSpectralIndex)
    assert ckd.w == w
    assert ckd.g == g
    assert ckd.formatted_repr == f"{w:g~P}:{g:g}"


def test_spectral_index_new_invalid():
    """Test the ``SpectralIndex.new()`` factory method with invalid arguments."""
    
    # mode = "invalid"

    with pytest.raises(ModeError):
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


def test_spectral_index_from_dict():
    """Test the ``SpectralIndex.from_dict()`` factory method."""
    
    # Monochromatic spectral index
    w = 1.0 * ureg.nm
    mono = SpectralIndex.from_dict({"mode": "mono", "w": w})
    assert isinstance(mono, MonoSpectralIndex)
    assert mono.w == w
    assert mono.formatted_repr == f"{w:g~P}"
    
    # CKD spectral index
    w = 1.0 * ureg.nm
    g = 0.5
    ckd = SpectralIndex.from_dict({"mode": "ckd", "w": w, "g": g})
    assert isinstance(ckd, CKDSpectralIndex)
    assert ckd.w == w
    assert ckd.g == g
    assert ckd.formatted_repr == f"{w:g~P}:{g:g}"


def test_spectral_index_convert():
    """Test the ``SpectralIndex.convert()`` method."""
    
    # Monochromatic spectral index
    w = 1.0 * ureg.nm
    mono = SpectralIndex.new(mode="mono", w=w)
    value = {"mode": "mono", "w": w}
    assert mono == SpectralIndex.convert(value)
    
    # CKD spectral index
    w = 1.0 * ureg.nm
    g = 0.5
    ckd = SpectralIndex.new(mode="ckd", w=w, g=g)
    value = {"mode": "ckd", "w": w, "g": g}
    assert ckd == SpectralIndex.convert(value)
