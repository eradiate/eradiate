import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.measure import HemisphericalDistantMeasure
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [
        {},
        dict(target=[0, 0, 0]),
        dict(target={"type": "rectangle", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}),
    ],
    ids=[
        "no_args",
        "target_point",
        "target_rectangle",
    ],
)
def test_hemispherical_distant_construct(modes_all_double, tested):
    measure = HemisphericalDistantMeasure(**tested)
    check_scene_element(measure, mi.Sensor)


def test_hemispherical_distant_viewing_angles(mode_mono):
    # Test default constructor
    d = HemisphericalDistantMeasure(film_resolution=(2, 2))
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
