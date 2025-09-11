import mitsuba as mi
import pytest

from eradiate.scenes.measure import CountMeasure
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {
            "voxel_resolution": [1, 1, 1],
            "bounding_box": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        },
        {
            "voxel_resolution": [1, 1, 1],
        },
    ],
    ids=[
        "no_args",
        "all",
        "no_bbox",
    ],
)
def test_count(modes_all_double, tested):
    measure = CountMeasure(**tested)
    check_scene_element(measure, mi.Sensor)
