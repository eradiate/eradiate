import mitsuba as mi
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import LambertianBSDF
from eradiate.scenes.core import traverse
from eradiate.scenes.spectra import UniformSpectrum


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"reflectance": 0.5}],
    ids=["noargs", "args"],
)
def test_lambertian_construct(modes_all, kwargs):
    # Default constructor
    bsdf = LambertianBSDF(**kwargs)
    assert isinstance(bsdf.reflectance, UniformSpectrum)


def test_lambertian_kernel_dict(modes_all_double):
    bsdf = LambertianBSDF()
    template, _ = traverse(bsdf)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx=ctx)
    assert isinstance(mi.load_dict(kernel_dict), mi.BSDF)
