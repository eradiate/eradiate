import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import RTLSBSDF
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"f_iso": 0.3, "f_vol": 1.4, "f_geo": -0.23},
        {
            "f_iso": 0.3,
            "f_vol": {"type": "uniform", "value": 0.3},
            "f_geo": {
                "type": "interpolated",
                "wavelengths": [300.0, 800.0],
                "values": [-0.23, 0.23],
            },
            "h": 0.2,
        },
    ],
    ids=["noargs", "uniform", "mixed"],
)
def test_rpv_construct(modes_all, kwargs):
    # Default constructor
    assert RTLSBSDF(**kwargs)


def test_rtls(modes_all_double):
    bsdf = RTLSBSDF(
        f_iso=0.3,
        f_geo={"type": "uniform", "value": 0.3},
        f_vol={
            "type": "interpolated",
            "wavelengths": [300.0, 800.0],
            "values": [-0.23, 0.23],
        },
        h=0.2,
        r=0.1,
        b=0.97
    )

    check_scene_element(bsdf, mi.BSDF)
