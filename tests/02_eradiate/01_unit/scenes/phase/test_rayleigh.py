import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import NodeSceneElement, traverse
from eradiate.scenes.phase import PhaseFunction, RayleighPhaseFunction
from eradiate.test_tools.types import check_node_scene_element, check_type


def test_rayleigh_type():
    check_type(
        RayleighPhaseFunction,
        expected_mro=[PhaseFunction, NodeSceneElement],
        expected_slots=[],
    )


def test_rayleigh(modes_all_double):
    phase = RayleighPhaseFunction()
    check_node_scene_element(phase, mi.PhaseFunction)
