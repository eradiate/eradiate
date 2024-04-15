import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.quad import Quad, QuadType
from eradiate.spectral import Bin, BinSet
from eradiate.spectral.ckd import (
    QuadSpec,
    QuadSpecErrorThreshold,
    QuadSpecFixed,
    QuadSpecMinError,
)


def test_ckd_bin():
    """
    Unit tests for :class:`eradiate.ckd.Bin`.
    """
    quad = Quad.gauss_legendre(16)

    # Construct a bin
    bin = Bin(wmin=500.0, wmax=510.0, quad=quad)

    # Scalar values are correctly converted to config default units
    assert bin.wmin == 500.0 * ureg.nm

    # Wrong bound ordering raises
    with pytest.raises(ValueError):
        Bin(wmin=510.0, wmax=500.0, quad=quad)


def test_bin_set_arange():
    binset = BinSet.arange(
        500 * ureg.nm, 600 * ureg.nm, 10 * ureg.nm, quad=Quad.gauss_legendre(2)
    )
    result = binset.wavelengths
    expected = [505, 515, 525, 535, 545, 555, 565, 575, 585, 595] * ureg.nm
    assert np.allclose(result, expected)


def test_quad_spec_default():
    assert QuadSpec.default() == QuadSpecFixed(quad_type=QuadType.GAUSS_LEGENDRE, n=16)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            {"type": "fixed", "n": 8, "quad_type": "gauss_lobatto"},
            QuadSpecFixed(8, "gauss_lobatto"),
        ),
        (
            {"type": "minimize", "nmax": 8},
            QuadSpecMinError(8),
        ),
        (
            {"type": "threshold", "nmax": 8, "threshold": 0.01},
            QuadSpecErrorThreshold(0.01, 8),
        ),
        ({"type": "invalid"}, ValueError),
        ({}, ValueError),
    ],
)
def test_quad_spec_from_dict(value, expected):
    """
    Unit tests for :class:`.QuadSpec`.
    """
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            QuadSpec.from_dict(value)

    else:
        quad_spec = QuadSpec.from_dict(value)
        assert quad_spec == expected
