from eradiate.scenes.builder.films import HDRFilm
from eradiate.scenes.builder.samplers import Independent
from eradiate.scenes.builder.sensors import Distant, Perspective, RadianceMeter
from eradiate.scenes.builder.transforms import LookAt, Transform


def test_perspective(variant_scalar_mono):
    # Basic init
    s = Perspective()
    assert s.to_xml() == """<sensor type="perspective"/>"""
    s.instantiate()

    # Init with params
    s = Perspective(
        sampler=Independent(sample_count=128),
        film=HDRFilm(pixel_format="luminance"),
        to_world=Transform([
            LookAt(origin=[10, 50, -800], target=[0, 0, 0], up=[0, 1, 0])
        ]),
    )
    assert s.to_xml() == \
        '<sensor type="perspective">' \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="128"/>' \
        '</sampler>' \
        '<film type="hdrfilm">' \
        '<string name="pixel_format" value="luminance"/>' \
        '</film>' \
        '<transform name="to_world">' \
        '<lookat origin="10, 50, -800" target="0, 0, 0" up="0, 1, 0"/>' \
        '</transform>' \
        '</sensor>'
    s.instantiate()


def test_radiancemeter(variant_scalar_mono):
    # Basic init
    s = RadianceMeter()
    assert s.to_xml() == \
        '<sensor type="radiancemeter">' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '</film>' \
        '</sensor>'

    # Init with params
    s = RadianceMeter(
        origin=[0, 0, 0],
        direction=[0, 0, 1],
        sampler=Independent(sample_count=128),
        film=HDRFilm(pixel_format="luminance"),
    )
    assert s.to_xml() == \
        '<sensor type="radiancemeter">' \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="128"/>' \
        '</sampler>' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '<string name="pixel_format" value="luminance"/>' \
        '</film>' \
        '<point name="origin" value="0, 0, 0"/>' \
        '<vector name="direction" value="0, 0, 1"/>' \
        '</sensor>'
    s.instantiate()


def test_distant(variant_scalar_mono):
    # Basic init
    s = Distant()
    assert s.to_xml() == \
        '<sensor type="distant">' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '</film>' \
        '</sensor>'

    # Init with params
    s = Distant(
        direction=[0, 0, -1],
        target=[0, 0, 0],
        sampler=Independent(sample_count=128),
        film=HDRFilm(pixel_format="luminance"),
    )
    assert s.to_xml() == \
        '<sensor type="distant">' \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="128"/>' \
        '</sampler>' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '<string name="pixel_format" value="luminance"/>' \
        '</film>' \
        '<vector name="direction" value="0, 0, -1"/>' \
        '<point name="target" value="0, 0, 0"/>' \
        '</sensor>'
    s.instantiate()
