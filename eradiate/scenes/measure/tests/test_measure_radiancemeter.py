import pytest

from eradiate.contexts import KernelDictContext
from eradiate.exceptions import UnsupportedModeError
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
    assert KernelDict.from_elements(s, ctx=ctx).load() is not None


def test_radiancemeter_unsupported_mode(modes_all_rgb):
    with pytest.raises(UnsupportedModeError):
        s = RadiancemeterMeasure()
