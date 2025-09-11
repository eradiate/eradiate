import mitsuba as mi
import pytest

from eradiate.scenes.measure import VoxelFluxMeasure
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "tested",
    [
        {},
        {
            "voxel_resolution": [1, 2, 3],
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
def test_voxel_flux(modes_all_double, tested):
    measure = VoxelFluxMeasure(**tested)
    check_scene_element(measure, mi.Sensor)

    film_res = measure.film_resolution
    vox_res = measure.voxel_resolution
    assert film_res == (2, 3, vox_res[0] + 1, vox_res[1] + 1, vox_res[2] + 1, 1)
