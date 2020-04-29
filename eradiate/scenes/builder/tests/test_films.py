from eradiate.scenes.builder.films import *
from eradiate.scenes.builder.rfilters import Box


def test_hdrfilm(variant_scalar_mono):
    # Default init
    f = HDRFilm()
    assert f.to_xml() == """<film type="hdrfilm"/>"""
    f.instantiate()

    # Init with filter
    f = HDRFilm(rfilter=Box())
    assert f.to_xml() == \
        '<film type="hdrfilm">' \
        '<rfilter type="box"/>' \
        '</film>'
    f.instantiate()

    # Init with some params
    f = HDRFilm(high_quality_edges=True, rfilter=Box())
    assert f.to_xml() == \
        '<film type="hdrfilm">' \
        '<boolean name="high_quality_edges" value="true"/>' \
        '<rfilter type="box"/>' \
        '</film>'
    f.instantiate()
