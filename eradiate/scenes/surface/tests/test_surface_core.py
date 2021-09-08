import pytest

from eradiate import unit_registry as ureg
from eradiate.attrs import AUTO
from eradiate.contexts import KernelDictContext
from eradiate.exceptions import ConfigWarning, OverriddenValueWarning
from eradiate.scenes.surface import Surface


class MyBlackSurface(Surface):
    """A basic implementation for the sake of testing."""

    def bsdfs(self, ctx: KernelDictContext):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.0},
            }
        }


def test_kernel_width(mode_mono):
    obj = MyBlackSurface()

    # Size value is appropriately used as kernel object size
    ctx = KernelDictContext()
    obj.width = 1.0 * ureg.m
    assert obj.kernel_width(ctx) == 1.0 * ureg.m

    # Auto size yields 100 km kernel width
    obj.width = AUTO
    assert obj.kernel_width(ctx) == 100.0 * ureg.km

    # Override constrains auto size
    obj.width = AUTO
    ctx = KernelDictContext(override_scene_width=100.0 * ureg.m)
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
    obj.width = AUTO
    with pytest.warns(ConfigWarning):
        obj_scaled = obj.scaled(2.0)
    assert obj_scaled is not obj
    assert obj_scaled.width is AUTO


def test_default_altitude():
    """
    Sets the altitude to 0.0 km by default.
    """
    obj = MyBlackSurface()
    assert obj.altitude == 0.0 * ureg.km


def test_negative_altitude():
    """
    Raises when altitude is set to negative value.
    """
    obj = MyBlackSurface()

    with pytest.raises(ValueError):
        obj.altitude = -1 * ureg.km
