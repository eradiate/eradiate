import mitsuba as mi

from eradiate.scenes.bsdfs import BSDF, BlackBSDF
from eradiate.scenes.core import NodeSceneElement
from eradiate.test_tools.types import check_node_scene_element, check_type


def test_black_type():
    check_type(
        BlackBSDF,
        expected_mro=[BSDF, NodeSceneElement],
        expected_slots=[],
    )


def test_black_constructor(modes_all):
    # Default constructor
    assert BlackBSDF()


def test_black_kernel_dict(modes_all_double):
    b = BlackBSDF()
    mi_obj, mi_params = check_node_scene_element(b, mi.BSDF)
    assert mi_params["reflectance.value"] == 0.0
