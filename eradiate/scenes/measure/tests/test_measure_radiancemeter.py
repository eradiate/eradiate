from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._radiancemeter import RadiancemeterMeasure


def test_radiancemeter(mode_mono):
    # Instantiation succeeds
    s = RadiancemeterMeasure()

    # Film is appropriately set
    assert s._film_dicts() == [
        {
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            }
        }
    ]

    # The kernel dict can be instantiated
    ctx = KernelDictContext()
    assert KernelDict.new(s, ctx=ctx).load() is not None
