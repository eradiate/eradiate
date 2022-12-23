import mitsuba as mi

from eradiate.scenes.phase import RayleighPhaseFunction
from eradiate.test_tools.types import check_scene_element


def test_rayleigh(modes_all_double):
    phase = RayleighPhaseFunction()
    check_scene_element(phase, mi.PhaseFunction)
