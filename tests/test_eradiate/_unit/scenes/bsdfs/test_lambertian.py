from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import LambertianBSDF


def test_lambertian(modes_all_double):
    ctx = KernelDictContext()

    # Default constructor
    bsdf = LambertianBSDF()
    assert bsdf.kernel_dict(ctx).load()

    # Constructor with arguments
    bsdf = LambertianBSDF(reflectance={"type": "uniform", "value": 0.3})
    assert bsdf.kernel_dict(ctx).load()
