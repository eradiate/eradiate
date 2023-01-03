import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.measure import Target, TargetPoint, TargetRectangle


def test_target_origin(modes_all):
    # TargetPoint: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetPoint([0, 0, 0])
        assert t.xyz.units == ureg.km

    with pytest.raises(ValueError):
        TargetPoint(0)

    # TargetPoint: check kernel item
    with ucc.override({"length": "km"}), uck.override({"length": "m"}):
        t = TargetPoint([1, 2, 0])
        assert dr.allclose(t.kernel_item(), [1000, 2000, 0])

    # TargetRectangle: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetRectangle(0, 1, 0, 1)
        assert t.xmin == 0.0 * ureg.km
        assert t.xmax == 1.0 * ureg.km
        assert t.ymin == 0.0 * ureg.km
        assert t.ymax == 1.0 * ureg.km

    with ucc.override({"length": "m"}):
        t = TargetRectangle(0, 1, 0, 1)
        assert t.xmin == 0.0 * ureg.m
        assert t.xmax == 1.0 * ureg.m
        assert t.ymin == 0.0 * ureg.m
        assert t.ymax == 1.0 * ureg.m

    with pytest.raises(ValueError):
        TargetRectangle(0, 1, "a", 1)

    with pytest.raises(ValueError):
        TargetRectangle(0, 1, 1, -1)

    # TargetRectangle: check kernel item
    t = TargetRectangle(-1, 1, -1, 1)

    with uck.override({"length": "mm"}):  # Tricky: we can't compare transforms directly
        kernel_item = t.kernel_item()["to_world"]
        assert dr.allclose(
            kernel_item.transform_affine(mi.Point3f(-1, -1, 0)), [-1000, -1000, 0]
        )
        assert dr.allclose(
            kernel_item.transform_affine(mi.Point3f(1, 1, 0)), [1000, 1000, 0]
        )
        assert dr.allclose(
            kernel_item.transform_affine(mi.Point3f(1, 1, 42)), [1000, 1000, 42]
        )

    # Factory: basic test
    with ucc.override({"length": "m"}):
        t = Target.new("point", xyz=[1, 1, 0])
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = Target.new("rectangle", 0, 1, 0, 1)
        assert isinstance(t, TargetRectangle)

    # Converter: basic test
    with ucc.override({"length": "m"}):
        t = Target.convert({"type": "point", "xyz": [1, 1, 0]})
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = Target.convert([1, 1, 0])
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        with pytest.raises(ValueError):
            Target.convert({"xyz": [1, 1, 0]})
