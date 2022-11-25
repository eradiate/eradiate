import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.ckd import Bin
from eradiate.ckd_next import Bin as BinNext
from eradiate.contexts import (
    CKDSpectralContext,
    CKDSpectralContextNext,
    KernelDictContext,
    MonoSpectralContext,
    SpectralContext,
)
from eradiate.quad import Quad


def test_spectral_context_new(modes_all):
    """
    Unit tests for :meth:`SpectralContext.new`.
    """

    # Empty call to new() should yield the appropriate SpectralContext instance
    if eradiate.mode().is_mono:
        assert isinstance(SpectralContext.new(), MonoSpectralContext)
    elif eradiate.mode().is_ckd:
        assert isinstance(SpectralContext.new(), CKDSpectralContext)
    else:
        # All modes must have a context: this test fails if the mode has no
        # associated spectral context
        assert False


def test_mono_spectral_context(mode_mono):
    """
    Unit tests for :class:`MonoSpectralContext`.
    """

    ctx = SpectralContext.new(wavelength=550.0)

    # Wavelength is stored as a Pint quantity
    assert ctx.wavelength == 550.0 * ureg.nm

    # Index is a plain float
    assert ctx.spectral_index == 550.0

    # Index string repr is nice and compact
    assert ctx.spectral_index_formatted == "550 nm"


def test_ckd_spectral_context(mode_ckd):
    """
    Unit tests for :class:`CKDSpectralContext`.
    """
    quad = Quad.gauss_legendre(16)
    bin = Bin.convert({"id": "510", "wmin": 505.0, "wmax": 515.0, "quad": quad})
    ctx = SpectralContext.new(bindex=bin.bindexes[8])

    # Wavelength is equal to bin central wavelength
    assert ctx.wavelength == 510.0 * ureg.nm

    # Index is a (str, int) pair
    assert ctx.spectral_index == ("510", 8)

    # Index string repr is a compact "{bin_id}:{quad_point_index}" string
    assert ctx.spectral_index_formatted == "510:8"


def test_ckd_spectral_context_next(mode_ckd):
    """
    Unit tests for :meth:`CKDSpectralContextNext`.
    """
    quad = Quad.gauss_legendre(8)
    bin = BinNext.convert({"wmin": 505.0, "wmax": 515.0, "quad": quad})
    ctx = CKDSpectralContextNext(bing=bin.bings[7])

    assert ctx.wavelength == 510.0 * ureg.nm

    # index is a (center wavelength, g-point) pair
    assert ctx.spectral_index == (bin.wcenter, quad.nodes[7])

    # Index string representation is "{bin center wavelength}:{g-point}" string
    assert ctx.spectral_index_formatted == f"{bin.wcenter:.2f~}:{quad.nodes[7]:.2f}"


def test_kernel_dict_context_construct(modes_all_double):
    # A default context can be instantiated without argument
    assert KernelDictContext()


def test_kernel_dict_context_evolve(mode_mono_double):
    ctx_1 = KernelDictContext()

    # Evolving the context adds additional data to its dynamic component
    with pytest.warns(UserWarning):
        ctx_2 = ctx_1.evolve(foo="bar")
    assert ctx_2.foo == "bar"
    assert ctx_1 is not ctx_2
