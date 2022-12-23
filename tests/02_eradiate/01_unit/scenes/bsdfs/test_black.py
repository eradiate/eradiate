import mitsuba as mi

from eradiate.scenes.bsdfs import BlackBSDF
from eradiate.test_tools.types import check_scene_element


def test_black_constructor(modes_all):
    # Default constructor
    assert BlackBSDF()


def test_black_kernel_dict(modes_all_double):
    b = BlackBSDF()
    mi_obj, mi_params = check_scene_element(b, mi.BSDF)
    assert mi_params["reflectance.value"] == 0.0
