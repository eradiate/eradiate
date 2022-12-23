import mitsuba as mi
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.measure import PerspectiveCameraMeasure
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested, expected",
    [
        ({}, None),
        (
            {"origin": [0, 0, 0], "target": [0, 0, 0], "up": [0, 0, 1]},
            ValueError,
        ),
        (
            {"origin": [1, 1, 1], "target": [1, 1, 1], "up": [0, 0, 1]},
            ValueError,
        ),
        (
            {"origin": [-1, 0.5, 1.5], "target": [-1, 0.5, 1.5], "up": [0, 0, 1]},
            ValueError,
        ),
        (
            {"origin": [0, 1, 0], "target": [1, 0, 0], "up": [1, -1, 0]},
            ValueError,
        ),
    ],
    ids=[
        "no_args",
        "same_origin_target_1",  # origin and target cannot be the same
        "same_origin_target_2",
        "same_origin_target_3",
        "same_up_direction",  # up and viewing direction cannot be the same
    ],
)
def test_perspective_construct(mode_mono, tested, expected):
    # Constructor
    if expected is None:
        measure = PerspectiveCameraMeasure(**tested)
        check_scene_element(measure, mi.Sensor)

    elif issubclass(expected, ValueError):
        with pytest.raises(expected):
            PerspectiveCameraMeasure(**tested)

    else:
        RuntimeError("unhandled expected value")


def test_perspective_medium(mode_mono):
    measure = PerspectiveCameraMeasure()
    template, _ = traverse(measure)

    ctx1 = KernelDictContext()
    ctx2 = KernelDictContext(kwargs={"atmosphere_medium_id": "test_atmosphere"})

    kd1 = template.render(ctx=ctx1, drop=True)
    assert "medium" not in kd1

    kd2 = template.render(ctx=ctx2, drop=True)
    assert kd2["medium"] == {"type": "ref", "id": "test_atmosphere"}
