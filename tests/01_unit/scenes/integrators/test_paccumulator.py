import mitsuba as mi
import pytest

from eradiate.scenes.integrators import PAccumulatorIntegrator
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "min_depth": 1,
            "max_depth": 10,
            "rr_depth": 5,
            "periodic_box": {"center": [0, 0, 0], "edges": [10, 10, 10]},
        },
        {
            "min_depth": 1,
            "max_depth": 10,
            "rr_depth": 5,
        },
    ],
    ids=["noargs", "args", "args_no_pbox"],
)
def test_paccumulator_construct(modes_all, kwargs):
    # Construction succeed
    integrator = PAccumulatorIntegrator(**kwargs)
    check_scene_element(integrator, mi.Integrator)
