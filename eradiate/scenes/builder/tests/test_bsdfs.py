from eradiate.scenes.builder import *


def test_null(variant_scalar_mono):
    # Default init
    b = bsdfs.Null()
    assert b.to_xml() == """<bsdf type="null"/>"""
    b.instantiate()


def test_diffuse(variant_scalar_mono):
    # Default init
    b = bsdfs.Diffuse()
    assert b.to_xml() == """<bsdf type="diffuse"/>"""
    b.instantiate()

    # Init with reflectance
    b = bsdfs.Diffuse(reflectance=Spectrum(1.0))
    assert b.to_xml() == """<bsdf type="diffuse"><spectrum name="reflectance" value="1.0"/></bsdf>"""
    b.instantiate()


def test_rough_dielectric(variant_scalar_mono):
    # Default init
    b = bsdfs.RoughDielectric()
    assert b.to_xml() == """<bsdf type="roughdielectric"/>"""
    b.instantiate()

    # Init with parameters
    b = bsdfs.RoughDielectric(
        distribution="beckmann",
        alpha=0.1,
        int_ior="bk7",
        ext_ior="air"
    )
    assert b.to_xml() == \
        '<bsdf type="roughdielectric">' \
        '<string name="int_ior" value="bk7"/>' \
        '<string name="ext_ior" value="air"/>' \
        '<string name="distribution" value="beckmann"/>' \
        '<float name="alpha" value="0.1"/>' \
        '</bsdf>'
    b.instantiate()
