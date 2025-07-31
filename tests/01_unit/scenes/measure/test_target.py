import drjit as dr
import numpy as np
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.measure import Target, TargetPoint, TargetRectangle


def test_target_point(mode_mono):
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


@pytest.mark.parametrize(
    "kwargs, config_units, kernel_units, expected",
    [
        (
            {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1, "z": 0},
            "m",
            "m",
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
        (
            {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1},
            "m",
            "m",
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
        (
            {
                "size_x": 2,
                "size_y": 4,
                "xyz": [0, 0, 1],
                "n": [0, 0, 1],
                "up": [0, 1, 0],
            },
            "m",
            "m",
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ],
        ),
        (
            {
                "size_x": 1,
                "size_y": 1,
                "xyz": [0, 0, 1],
                "n": [0, 0, 1],
                "up": [0, 1, 0],
            },
            "km",
            "mm",
            [
                [5e5, 0, 0, 0],
                [0, 5e5, 0, 0],
                [0, 0, 1, 1e6],
                [0, 0, 0, 1],
            ],
        ),
        (
            {
                "to_world": [
                    [1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ]
            },
            "m",
            "km",
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
        ),
    ],
    ids=["bounds", "bounds_no_z", "normal", "normal_scaled", "to_world"],
)
def test_target_rectangle(mode_mono, kwargs, config_units, kernel_units, expected):
    with ucc.override(length=config_units), uck.override(length=kernel_units):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                TargetRectangle(**kwargs)
                return

        else:
            t = TargetRectangle(**kwargs)
            np.testing.assert_allclose(t.to_world.matrix.numpy(), expected)


def test_target_factory(mode_mono):
    # Factory: basic test
    with ucc.override({"length": "m"}):
        t = Target.new("point", xyz=[1, 1, 0])
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = Target.new("rectangle", xmin=0, xmax=1, ymin=0, ymax=1, z=0)
        assert isinstance(t, TargetRectangle)

    # Converter: basic test
    with ucc.override({"length": "km"}):
        t = Target.convert({"type": "point", "xyz": [1, 1, 0]})
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, [1000, 1000, 0] * ureg.m)

        t = Target.convert([1, 1, 0])
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, [1000, 1000, 0] * ureg.m)

        with pytest.raises(ValueError):
            Target.convert({"xyz": [1, 1, 0]})
