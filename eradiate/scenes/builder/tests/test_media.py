import pytest

from eradiate.scenes.builder.media import *
from eradiate.scenes.builder.phase import Isotropic
from eradiate.scenes.builder.spectra import Spectrum


def test_homogeneous_init(variant_scalar_mono):
    """ Instance creation """

    # Default constructor
    m = Homogeneous()
    assert m.to_xml() == \
        '<medium type="homogeneous">' \
        '<spectrum name="sigma_t" value="1e-05"/>' \
        '<spectrum name="albedo" value="0.99"/>'\
        '</medium>'
    m.instantiate()

    # Construct with full specification
    m = Homogeneous(
        sigma_t=Spectrum(0.2),
        albedo=Spectrum(0.35)
    )

    assert m.to_xml() == \
        '<medium type="homogeneous">' \
        '<spectrum name="sigma_t" value="0.2"/>' \
        '<spectrum name="albedo" value="0.35"/>' \
        '</medium>'
    m.instantiate()

    # Construct from collision coefficients
    m = Homogeneous.from_collision_coefficients(
        sigma_s=Spectrum(0.01),
        sigma_a=Spectrum(0.0)
    )
    assert m.sigma_s.value == 0.01
    assert m.sigma_a.value == 0.0
    assert m.to_xml() == \
        '<medium type="homogeneous">' \
        '<spectrum name="sigma_t" value="0.01"/>' \
        '<spectrum name="albedo" value="1.0"/>'\
        '</medium>'
    m.instantiate()

    # Construct from collision coefficients and phase function
    m = Homogeneous.from_collision_coefficients(
        phase=Isotropic(),
        sigma_s=Spectrum(0.01),
        sigma_a=Spectrum(0.0)
    )
    assert m.to_xml() == \
        '<medium type="homogeneous">' \
        '<phase type="isotropic"/>' \
        '<spectrum name="sigma_t" value="0.01"/>' \
        '<spectrum name="albedo" value="1.0"/>'\
        '</medium>'
    m.instantiate()

    # Raise if an unexpected argument is passed
    with pytest.raises(TypeError):
        m = Homogeneous.from_collision_coefficients(
            phase=Isotropic(),
            sigma_s=Spectrum(0.01),
            sigma_a=Spectrum(0.0),
            wrong=True,
        )
    with pytest.raises(TypeError):
        m = Homogeneous.from_collision_coefficients(
            phase=Isotropic(),
            sigma_s=Spectrum(0.01),
            sigma_a=Spectrum(0.0),
            sigma_t="whatever",
        )
