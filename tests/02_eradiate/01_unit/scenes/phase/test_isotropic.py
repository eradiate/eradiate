import mitsuba as mi

from eradiate.scenes.core import NodeSceneElement
from eradiate.scenes.phase import IsotropicPhaseFunction, PhaseFunction
from eradiate.test_tools.types import check_node_scene_element, check_type


def test_isotropic_type():
    check_type(
        IsotropicPhaseFunction,
        expected_mro=[PhaseFunction, NodeSceneElement],
        expected_slots=[],
    )


def test_isotropic(modes_all_double):
    # Default constructor
    phase = IsotropicPhaseFunction()
    check_node_scene_element(phase, mi.PhaseFunction)
