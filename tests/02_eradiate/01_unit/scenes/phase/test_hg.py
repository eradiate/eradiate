import mitsuba as mi
import pytest

from eradiate.scenes.phase import HenyeyGreensteinPhaseFunction, PhaseFunction
from eradiate.test_tools.types import check_node_scene_element, check_type


def test_hg_type():
    check_type(
        HenyeyGreensteinPhaseFunction,
        expected_mro=[HenyeyGreensteinPhaseFunction, PhaseFunction],
        expected_slots=[],
    )


@pytest.mark.parametrize("kwargs", [{}, {"g": 0.25}], ids=["noargs", "args"])
def test_hg(modes_all_double, kwargs):
    phase = HenyeyGreensteinPhaseFunction(**kwargs)
    check_node_scene_element(phase, mi.PhaseFunction)
