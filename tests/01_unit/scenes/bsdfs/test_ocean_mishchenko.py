import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import OceanMishchenkoBSDF
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "wind_speed": 2.0,
            "eta": 1.39,
            "k": 0.0,
            "ext_ior": 1.00027,
        },
        {"eta": 1.34, "ext_ior": 1.0},
    ],
    ids=["noargs", "uniform", "mixed"],
)
def test_ocean_mishchenko_construct(modes_all, kwargs):
    # Default constructor
    assert OceanMishchenkoBSDF(**kwargs)


def test_ocean_mishchenko(modes_all_double):
    bsdf = OceanMishchenkoBSDF(
        wind_speed=15.0,
        eta=1.33,
        k=0.01,
        ext_ior=1.02,
    )

    check_scene_element(bsdf, mi.BSDF)
