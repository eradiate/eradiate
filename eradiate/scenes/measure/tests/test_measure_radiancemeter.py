from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._radiancemeter import RadiancemeterMeasure


def test_radiancemeter(mode_mono):
    # Instantiation succeeds
    s = RadiancemeterMeasure()

    # Film is appropropriately set
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

    # The kernel dict can be instantiated
    assert KernelDict.empty().add(s).load() is not None
