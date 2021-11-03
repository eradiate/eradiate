from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.surface import CheckerboardSurface


def test_checkerboard(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    s = CheckerboardSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.from_elements(s, ctx=ctx)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    s = CheckerboardSurface(
        width=1000.0, reflectance_a=0.5, reflectance_b=0.1, scale_pattern=1.5
    )

    # Check if produced scene can be instantiated
    assert KernelDict.from_elements(s, ctx=ctx).load() is not None
