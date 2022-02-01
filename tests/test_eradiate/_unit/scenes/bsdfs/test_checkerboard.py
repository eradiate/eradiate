from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import CheckerboardBSDF


def test_checkerboard(modes_all_double):
    ctx = KernelDictContext()

    # Default constructor
    checkerboard = CheckerboardBSDF()
    assert checkerboard.kernel_dict(ctx).load()

    # Constructor with arguments
    checkerboard = CheckerboardBSDF(
        reflectance_a=0.5, reflectance_b=0.1, scale_pattern=1.5
    )
    assert checkerboard.kernel_dict(ctx).load()
