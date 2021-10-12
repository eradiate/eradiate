from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._multi_radiancemeter import MultiRadiancemeterMeasure


def test_multi_radiancemeter_noargs(mode_mono):
    # Instantiation succeeds
    s = MultiRadiancemeterMeasure()

    # Film is appropriately set
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
    ctx = KernelDictContext()
    assert KernelDict.from_elements(s, ctx=ctx).load() is not None


def test_multi_radiancemeter_args(mode_mono):
    # Instantiation succeeds
    s = MultiRadiancemeterMeasure(
        origins=[[0, 0, 0]] * 3, directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    # Film is appropriately set
    assert s._film_dicts() == [
        {
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 3,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            }
        }
    ]

    # The kernel dict can be instantiated
    ctx = KernelDictContext()
    assert KernelDict.from_elements(s, ctx=ctx).load() is not None
