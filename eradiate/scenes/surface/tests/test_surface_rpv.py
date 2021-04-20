import numpy as np

from eradiate import unit_registry as ureg
from eradiate.scenes.core import KernelDict
from eradiate.contexts import KernelDictContext
from eradiate.scenes.surface import RPVSurface


def test_rpv(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    ls = RPVSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.new(ls, ctx=ctx)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = RPVSurface(width=ureg.Quantity(1000.0, "km"), rho_0=0.3, k=1.4, ttheta=-0.23)
    assert np.allclose(ls.width, ureg.Quantity(1e6, ureg.m))

    # Check if produced scene can be instantiated
    assert KernelDict.new(ls, ctx=ctx).load() is not None
