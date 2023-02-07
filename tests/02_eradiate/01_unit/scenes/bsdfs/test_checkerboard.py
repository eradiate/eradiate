import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import CheckerboardBSDF
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"reflectance_a": 0.5, "reflectance_b": 0.1, "scale_pattern": 1.5}],
    ids=["noargs", "args"],
)
def test_checkerboard_construct(modes_all, kwargs):
    # Default constructor
    assert CheckerboardBSDF(**kwargs)


def test_checkerboard_kernel_dict(modes_all_double):
    bsdf = CheckerboardBSDF()
    mi_wrapper = check_scene_element(bsdf, mi.BSDF)
    assert "reflectance.color0.value" in mi_wrapper.parameters.keys()
    assert "reflectance.color1.value" in mi_wrapper.parameters.keys()
