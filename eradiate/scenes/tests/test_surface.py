import numpy as np


from eradiate.scenes.core import KernelDict
from eradiate.scenes.surface import (
    LambertianSurface,
    RPVSurface,
    BlackSurface,
)
from eradiate import unit_registry as ureg


def test_lambertian(mode_mono):
    # Default constructor
    ls = LambertianSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = LambertianSurface(width=1000., reflectance={"type": "uniform", "value": .3})

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(ls).load() is not None


def test_rpv(mode_mono):
    # Default constructor
    ls = RPVSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = RPVSurface(
        width=ureg.Quantity(1000., "km"),
        rho_0=0.3,
        k=1.4,
        ttheta=-0.23
    )
    assert np.allclose(ls.width, ureg.Quantity(1e6, ureg.m))

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(ls).load() is not None


def test_black(mode_mono):
    # Default constructor
    bs = BlackSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(bs)
    assert kernel_dict.load() is not None

    # Check if the correct kernel dict is created
    ls = LambertianSurface(reflectance={"type": "uniform", "value": 0})

    assert KernelDict.empty().add(ls) == KernelDict.empty().add(bs)
