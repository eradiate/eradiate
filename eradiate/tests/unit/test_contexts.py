import eradiate
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate.ckd import Bin
from eradiate.contexts import CKDSpectralContext, MonoSpectralContext, SpectralContext
from eradiate.quad import Quad


def test_spectral_context_new(modes_all):
    """
    Unit tests for :meth:`SpectralContext.new`.
    """

    # Empty call to new() should yield the appropriate SpectralContext instance
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        assert isinstance(SpectralContext.new(), MonoSpectralContext)
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
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
