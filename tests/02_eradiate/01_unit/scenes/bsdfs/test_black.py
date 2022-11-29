import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import BlackBSDF
from eradiate.scenes.core import traverse


def test_black_constructor(modes_all):
    # Default constructor
    assert BlackBSDF()


def test_black_kernel_dict(modes_all_double):
    b = BlackBSDF()
    template, _ = traverse(b)
    kernel_dict = template.render(ctx=KernelDictContext())
    assert kernel_dict["reflectance"]["value"] == 0.0
    assert isinstance(mi.load_dict(kernel_dict), mi.BSDF)
