from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import BlackBSDF, LambertianBSDF


def test_black(modes_all_double):
    ctx = KernelDictContext()

    # Default constructor
    black = BlackBSDF()
    assert black.kernel_dict(ctx).load()

    # Check if the correct kernel dict is created
    lambertian = LambertianBSDF(reflectance={"type": "uniform", "value": 0})
    assert lambertian.kernel_dict(ctx) == black.kernel_dict(ctx)
