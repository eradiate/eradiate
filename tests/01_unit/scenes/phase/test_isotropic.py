import mitsuba as mi

from eradiate.scenes.phase import IsotropicPhaseFunction
from eradiate.test_tools.types import check_scene_element


def test_isotropic(modes_all_double):
    # Default constructor
    phase = IsotropicPhaseFunction()
    check_scene_element(phase, mi.PhaseFunction)
