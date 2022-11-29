import mitsuba as mi
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import CheckerboardBSDF
from eradiate.scenes.core import traverse


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"reflectance_a": 0.5, "reflectance_b": 0.1, "scale_pattern": 1.5}],
    ids=["noargs", "args"],
)
def test_checkerboard_construct(modes_all, kwargs):
    # Default constructor
    assert CheckerboardBSDF(**kwargs)


def test_checkerboard_kernel_dict(modes_all_double):
    c = CheckerboardBSDF()
    template, _ = traverse(c)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx=ctx)
    assert isinstance(mi.load_dict(kernel_dict), mi.BSDF)
