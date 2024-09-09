import mitsuba as mi

import eradiate
from eradiate.scenes.core import traverse
from eradiate.scenes.phase import RayleighPhaseFunction
from eradiate.test_tools.types import check_scene_element


def test_rayleigh(modes_all_double):
    phase = RayleighPhaseFunction()
    check_scene_element(phase, mi.PhaseFunction)

    kdict, _ = traverse(phase)
    if eradiate.mode().is_polarized:
        assert kdict.data["type"] == "rayleigh_polarized"
