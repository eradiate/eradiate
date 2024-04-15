import mitsuba as mi

from eradiate.scenes.bsdfs import BlackBSDF
from eradiate.test_tools.types import check_scene_element


def test_black_constructor(modes_all):
    # Default constructor
    assert BlackBSDF()


def test_black_kernel_dict(modes_all_double):
    b = BlackBSDF()
    mi_wrapper = check_scene_element(
        b, mi.BSDF, drop_parameters=False
    )  # Do not drop untracked parameters: the reflectance (which we want to check) is untracked
    assert mi_wrapper.parameters["reflectance.value"] == 0.0
