from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._radiancemeter import RadiancemeterMeasure


def test_radiancemeter(mode_mono):
    # Instantiation succeeds
    s = RadiancemeterMeasure()

    # Film is appropropriately set
    # fmt: off
    assert s._film_dicts() == [{
        "film": {
            "type": "hdrfilm",
            "height": 1,
            "width": 1,
            "pixel_format": "luminance",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        }
    }]
    # fmt: on

    # The kernel dict can be instantiated
    assert KernelDict.new(s).load() is not None
