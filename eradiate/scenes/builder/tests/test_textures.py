from eradiate.scenes.builder import *


def test_checkerboard(variant_scalar_mono):
    # Default init
    t = textures.Checkerboard()
    assert t.to_xml() == """<texture type="checkerboard"/>"""
    t.instantiate()
    
    # Init with parameters
    t = textures.Checkerboard(
        color0=Spectrum(1.),
        color1=Spectrum(0.),
        to_uv=Transform([Scale(value=[0., 1., 0.])])
    )
    assert t.to_xml() == \
        '<texture type="checkerboard">' \
        '<transform name="to_uv">' \
        '<scale value="0.0, 1.0, 0.0"/>' \
        '</transform>' \
        '<spectrum name="color0" value="1.0"/>' \
        '<spectrum name="color1" value="0.0"/>' \
        '</texture>'
    t.instantiate()

    # Init with parameters (nested textures)
    t = textures.Checkerboard(
        color0=Spectrum(1.),
        color1=textures.Checkerboard(),
        to_uv=Transform([Scale(value=[0., 1., 0.])])
    )
    assert t.to_xml() == \
        '<texture type="checkerboard">' \
        '<transform name="to_uv">' \
        '<scale value="0.0, 1.0, 0.0"/>' \
        '</transform>' \
        '<spectrum name="color0" value="1.0"/>' \
        '<texture name="color1" type="checkerboard"/>' \
        '</texture>'
    t.instantiate()
