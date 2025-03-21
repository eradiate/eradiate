import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import OceanGraspBSDF
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "wind_speed": 1.0,
            "eta": 1.33,
            "k": 0.0,
            "ext_ior": 1.0,
            "water_body_reflectance": 0.1,
        },
        {"eta": 1.34},
    ],
    ids=["noargs", "allargs", "mixedargs"],
)
def test_ocean_grasp_construct(modes_all, kwargs):
    # Default constructor
    bsdf = OceanGraspBSDF(**kwargs)
    check_scene_element(bsdf, mi.BSDF)
