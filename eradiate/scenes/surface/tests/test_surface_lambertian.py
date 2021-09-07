from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.surface import LambertianSurface


def test_lambertian(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    ls = LambertianSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.from_elements(ls, ctx=ctx)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = LambertianSurface(width=1000.0, reflectance={"type": "uniform", "value": 0.3})

    # Check if produced scene can be instantiated
    assert KernelDict.from_elements(ls, ctx=ctx).load() is not None
