import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._distant_flux import DistantFluxMeasure


def test_distant_flux_construct(modes_all):
    # Test default constructor
    d = DistantFluxMeasure()
    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # Test target support
    # -- Target a point
    d = DistantFluxMeasure(target=[0, 0, 0])
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # -- Target an axis-aligned rectangular patch
    d = DistantFluxMeasure(
        target={"type": "rectangle", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    )
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None


@pytest.mark.parametrize(
    ["direction", "frame"],
    [
        (
            [1, 0, 0],
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        ),
        (
            [0, 0, 1],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ),
    ],
)
def test_distant_flux_direction(modes_all, direction, frame):
    d = DistantFluxMeasure(direction=direction)
    ctx = KernelDictContext()
    to_world = onedict_value(d.kernel_dict(ctx))["to_world"]

    # The reference frame is rotated as expected
    assert dr.allclose(to_world.transform_affine(mi.Vector3f(1, 0, 0)), frame[0])
    assert dr.allclose(to_world.transform_affine(mi.Vector3f(0, 1, 0)), frame[1])
    assert dr.allclose(to_world.transform_affine(mi.Vector3f(0, 0, 1)), frame[2])


def test_distant_flux_viewing_angles(mode_mono):
    # Test default constructor
    d = DistantFluxMeasure(film_resolution=(2, 2))
    expected = (
        np.array(
            [
                [
                    (41.409622, 225),
                    (41.409622, 135),
                ],
                [
                    (41.409622, 315),
                    (41.409622, 45),
                ],
            ]
        )
        * ureg.deg
    )
    assert np.allclose(expected, d.viewing_angles)
