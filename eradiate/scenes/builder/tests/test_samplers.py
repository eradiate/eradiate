from eradiate.scenes.builder.samplers import Independent
from eradiate.scenes.builder.util import load


def test_independent(variant_scalar_mono):
    # Default init
    s = Independent()
    assert s.to_xml() == """<sampler type="independent"/>"""
    s.instantiate()

    # Set sample count
    s = Independent(sample_count=32)
    assert s.to_xml() == \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="32"/>' \
        '</sampler>'
    s.instantiate()
