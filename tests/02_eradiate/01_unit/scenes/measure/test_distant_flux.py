import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.core import traverse
from eradiate.scenes.measure import DistantFluxMeasure
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {"target": [0, 0, 0]},
        {"target": {"type": "rectangle", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}},
    ],
    ids=[
        "no_args",
        "target_point",
        "target_rectangle",
    ],
)
def test_distant_flux_construct(modes_all_double, tested):
    measure = DistantFluxMeasure(**tested)
    check_scene_element(measure, mi_cls=mi.Sensor)


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
def test_distant_flux_direction(modes_all_double, direction, frame):
    measure = DistantFluxMeasure(direction=direction)
    template, _ = traverse(measure)
    to_world = template["to_world"]

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
    assert np.allclose(d.viewing_angles, expected)
