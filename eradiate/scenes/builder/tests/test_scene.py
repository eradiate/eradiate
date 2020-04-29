import pytest

from eradiate.scenes.builder.base import Ref
from eradiate.scenes.builder.bsdfs import Diffuse
from eradiate.scenes.builder.emitters import Constant
from eradiate.scenes.builder.films import HDRFilm
from eradiate.scenes.builder.integrators import Direct
from eradiate.scenes.builder.scene import Scene
from eradiate.scenes.builder.sensors import Perspective
from eradiate.scenes.builder.shapes import Rectangle
from eradiate.scenes.builder.spectra import Spectrum


def test_scene(variant_scalar_mono):
    # Empty init
    s = Scene()
    with pytest.raises(ValueError):
        s.check()

    # Init with wrong type for shape sequence
    with pytest.raises(TypeError):
        Scene(shapes=["hello_world"])

    s = Scene()
    s.shapes.append("hello_world")
    with pytest.raises(TypeError):
        s.check()

    # Minimal init
    s = Scene(shapes=[Rectangle()])
    assert s.to_xml() == """<scene version="0.1.0"><shape type="rectangle"/></scene>"""
    s.instantiate()

    s = Scene(emitter=Constant())
    assert s.to_xml() == """<scene version="0.1.0"><emitter type="constant"/></scene>"""
    s.instantiate()

    # Init with nonempty sequence
    s = Scene(
        integrator=Direct(),
        sensor=Perspective(film=HDRFilm(pixel_format="luminance")),
        emitter=Constant(),
        shapes=[Rectangle(bsdf=Diffuse(reflectance=Spectrum(0.5)))]
    )
    assert s.to_xml(pretty_print=False) == \
        '<scene version="0.1.0">' \
        '<shape type="rectangle">' \
        '<bsdf type="diffuse">' \
        '<spectrum name="reflectance" value="0.5"/>' \
        '</bsdf>' \
        '</shape>' \
        '<emitter type="constant"/>' \
        '<sensor type="perspective">' \
        '<film type="hdrfilm">' \
        '<string name="pixel_format" value="luminance"/>' \
        '</film>' \
        '</sensor>' \
        '<integrator type="direct"/>' \
        '</scene>'
    s.instantiate()

    # Test BSDF referencing
    s = Scene(
        integrator=Direct(),
        sensor=Perspective(film=HDRFilm(pixel_format="luminance")),
        emitter=Constant(),
        bsdfs=[Diffuse(id="lambert", reflectance=Spectrum(0.5))],
        shapes=[Rectangle(bsdf=Ref("lambert"))]
    )
    assert s.to_xml(pretty_print=False) == \
           '<scene version="0.1.0">' \
           '<bsdf type="diffuse" id="lambert">' \
           '<spectrum name="reflectance" value="0.5"/>' \
           '</bsdf>' \
           '<shape type="rectangle">' \
           '<ref id="lambert"/>' \
           '</shape>' \
           '<emitter type="constant"/>' \
           '<sensor type="perspective">' \
           '<film type="hdrfilm">' \
           '<string name="pixel_format" value="luminance"/>' \
           '</film>' \
           '</sensor>' \
           '<integrator type="direct"/>' \
           '</scene>'
    s.instantiate()
