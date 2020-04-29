from eradiate.scenes.builder.emitters import Constant, Directional, Area
from eradiate.scenes.builder.spectra import Spectrum


def test_constant(variant_scalar_mono):
    # Empty init
    e = Constant()
    assert e.to_xml() == """<emitter type="constant"/>"""
    e.instantiate()
    # Init from spectrum
    e = Constant(radiance=Spectrum(1.))
    assert e.to_xml() == \
        '<emitter type="constant">' \
        '<spectrum name="radiance" value="1.0"/>' \
        '</emitter>'
    e.instantiate()


def test_directional(variant_scalar_mono):
    # Empty init
    e = Directional()
    assert e.to_xml() == \
        '<emitter type="directional">' \
        '<spectrum name="irradiance" value="1.0"/>' \
        '</emitter>'
    e.instantiate()
    # Init from spectrum and direction
    e = Directional(direction=[0, 0, -1], irradiance=Spectrum(2.))
    assert e.to_xml() == \
        '<emitter type="directional">' \
        '<vector name="direction" value="0, 0, -1"/>' \
        '<spectrum name="irradiance" value="2.0"/>' \
        '</emitter>'
    e.instantiate()


def test_area(variant_scalar_mono):
    # Empty init
    e = Area()
    assert e.to_xml() == \
        '<emitter type="area">' \
        '<spectrum name="radiance" value="1.0"/>' \
        '</emitter>'
    e.instantiate()
    # Init from spectrum
    e = Area(radiance=Spectrum(2.0))
    assert e.to_xml() == \
        '<emitter type="area">' \
        '<spectrum name="radiance" value="2.0"/>' \
        '</emitter>'
    e.instantiate()
