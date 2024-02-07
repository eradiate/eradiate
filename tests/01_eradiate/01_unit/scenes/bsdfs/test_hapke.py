import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import HapkeBSDF
from eradiate.test_tools.types import check_scene_element
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"w": 0.1, "b": 0.2, "c": 0.3, "theta": 40.0 * ureg.deg, "B_0": 0.5, "h": 0.6},
        {
            "w": 0.1,
            "b": 0.2,
            "c": {"type": "uniform", "value": 0.3},
            "theta": {
                "type": "interpolated",
                "wavelengths": [300.0, 800.0],
                "values": [10.0, 20.0],
            },
            "B_0": 0.5,
            "h": 0.6,
        },
    ],
    ids=["uniform", "mixed"],
)
def test_hapke_construct(modes_all_double, kwargs):
    bsdf = HapkeBSDF(**kwargs)

    assert bsdf
    check_scene_element(bsdf, mi.BSDF)
