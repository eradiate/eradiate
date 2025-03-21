import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import OceanLegacyBSDF
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "wind_speed": 1.0,
            "wind_direction": 15.0,
            "chlorinity": 19.0,
            "pigmentation": 0.3,
            "shadowing": True,
        },
        {"chlorinity": 19.0, "pigmentation": 0.3},
    ],
    ids=["noargs", "allargs", "mixedargs"],
)
def test_ocean_legacy_construct(modes_all, kwargs):
    # Default constructor
    assert OceanLegacyBSDF(**kwargs)


def test_ocean_legacy(modes_all_double):
    bsdf = OceanLegacyBSDF(
        wind_speed=15.0,
        wind_direction=42.0,
        chlorinity=19.0,
        pigmentation=0.3,
    )

    check_scene_element(bsdf, mi.BSDF)
