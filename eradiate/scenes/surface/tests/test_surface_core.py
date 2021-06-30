import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.exceptions import ConfigWarning, OverriddenValueWarning
from eradiate.scenes.surface import Surface


class MyBlackSurface(Surface):
    """A basic implementation for the sake of testing."""

    def bsdfs(self, ctx=None):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.0},
            }
        }


def test_kernel_width(mode_mono):
    obj = MyBlackSurface()

    # Size value is appropriately used as kernel object size
    obj.width = 1.0 * ureg.m
    assert obj.kernel_width() == 1.0 * ureg.m

    # Auto size yields 100 km kernel width
    obj.width = "auto"
    assert obj.kernel_width() == 100.0 * ureg.km

    # Override constrains auto size
    obj.width = "auto"
    ctx = KernelDictContext(override_surface_width=100.0 * ureg.m)
    assert obj.kernel_width(ctx) == 100.0 * ureg.m

    # Override with set value raises a warning
    obj.width = 1.0 * ureg.m
    with pytest.warns(OverriddenValueWarning):
        assert obj.kernel_width(ctx) == 100.0 * ureg.m


def test_scale(mode_mono):
    obj = MyBlackSurface()

    # Scaling a surface with a set value yields a scaled copy
    obj.width = 1.0 * ureg.m
    obj_scaled = obj.scaled(2.0)
    assert obj_scaled is not obj
    assert obj_scaled.width == 2.0 * obj.width

    # Scaling a surface with auto width returns an unmodified copy
    obj.width = "auto"
    with pytest.warns(ConfigWarning):
        obj_scaled = obj.scaled(2.0)
    assert obj_scaled is not obj
    assert obj_scaled.width == "auto"


def test_negative_altitude():
    obj = MyBlackSurface()

    with pytest.raises(ValueError):
        obj.altitude = -1 * ureg.km
