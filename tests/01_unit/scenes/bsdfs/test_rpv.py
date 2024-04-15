import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import RPVBSDF
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"rho_0": 0.3, "k": 1.4, "g": -0.23},
        {
            "rho_0": 0.3,
            "k": {"type": "uniform", "value": 0.3},
            "g": {
                "type": "interpolated",
                "wavelengths": [300.0, 800.0],
                "values": [-0.23, 0.23],
            },
            "rho_c": 0.2,
        },
    ],
    ids=["noargs", "uniform", "mixed"],
)
def test_rpv_construct(modes_all, kwargs):
    # Default constructor
    assert RPVBSDF(**kwargs)


def test_rpv(modes_all_double):
    bsdf = RPVBSDF(
        rho_0=0.3,
        k={"type": "uniform", "value": 0.3},
        g={
            "type": "interpolated",
            "wavelengths": [300.0, 800.0],
            "values": [-0.23, 0.23],
        },
        rho_c=0.2,
    )

    check_scene_element(bsdf, mi.BSDF)
