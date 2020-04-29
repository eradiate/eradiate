from eradiate.scenes.builder.rfilters import Box
from eradiate.scenes.builder.util import load


def test_box(variant_scalar_mono):
    # Default init
    f = Box()
    assert f.to_xml() == """<rfilter type="box"/>"""
    f.instantiate()
