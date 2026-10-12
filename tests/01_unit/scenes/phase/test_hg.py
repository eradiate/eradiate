import mitsuba as mi
import pytest

from eradiate.scenes.phase import HenyeyGreensteinPhaseFunction
from eradiate.testing.types import check_scene_element


@pytest.mark.parametrize("kwargs", [{}, {"g": 0.25}], ids=["noargs", "args"])
def test_hg(modes_all_double, kwargs):
    phase = HenyeyGreensteinPhaseFunction(**kwargs)
    check_scene_element(phase, mi.PhaseFunction)
