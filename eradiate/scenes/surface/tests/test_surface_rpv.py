import numpy as np

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.surface import RPVSurface


def test_rpv(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    surface = RPVSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.new(surface, ctx=ctx)
    assert kernel_dict.load() is not None

    # Construct from floats
    surface = RPVSurface(width=ureg.Quantity(1000.0, "km"), rho_0=0.3, k=1.4, g=-0.23)
    assert np.allclose(surface.width, ureg.Quantity(1e6, ureg.m))

    # Construct from mixed spectrum types
    surface = RPVSurface(
        width=ureg.Quantity(1000.0, "km"),
        rho_0=0.3,
        k={"type": "uniform", "value": 0.3},
        g={
            "type": "interpolated",
            "wavelengths": [300.0, 800.0],
            "values": [-0.23, 0.23],
        },
    )

    # Check if produced scene can be instantiated
    assert KernelDict.new(surface, ctx=ctx).load() is not None
