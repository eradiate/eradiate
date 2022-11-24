import pytest

from eradiate import unit_registry as ureg
from eradiate.ckd_next import Bin, Bing
from eradiate.quad import Quad


def test_ckd_bin(mode_ckd):
    """
    Unit tests for :class:`eradiate.ckd_next.Bin`.
    """
    quad = Quad.gauss_legendre(8)

    # Construct a bin
    bin = Bin(wmin=500.0, wmax=510.0, quad=quad)

    # Scalar values are correctly converted to config default units
    assert bin.wmin == 500.0 * ureg.nm
    assert bin.wmax == 510.0 * ureg.nm

    # Central wavelength is the mean of bounds
    assert bin.wcenter == 505.0 * ureg.nm

    # Width is the difference between bounds
    assert bin.width == 10.0 * ureg.nm

    # Wrong bound ordering raises
    with pytest.raises(ValueError):
        Bin(wmin=510.0, wmax=500.0, quad=quad)

    # Conversion from tuple is supported
    assert isinstance(Bin.convert((500.0, 510.0, quad)), Bin)

    # Conversion from dict is supported
    assert isinstance(Bin.convert(dict(wmin=500.0, wmax=510.0, quad=quad)), Bin)

def test_ckd_bing(mode_ckd):
    """
    Unit tests for :class:`eradiate.ckd_next.Bing`.
    """
    quad = Quad.gauss_legendre(8)
    bin = Bin(wmin=500.0, wmax=510.0, quad=quad)

    # Construct a bing
    bing = Bing(bin=bin, g=0.5)

    assert bing.bin == bin
    assert bing.g == 0.5