import numpy as np


from eradiate.scenes.core import KernelDict
from eradiate.scenes.lithosphere import LambertianSurface, RPVSurface
from eradiate.util.units import ureg


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
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = RPVSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = RPVSurface(
        width=ureg.Quantity(1000., "km"),
        width_units=ureg.m,
        rho_0=0.3,
        k=1.4,
        ttheta=-0.23
    )
    assert np.allclose(ls.width, 1e6)  # Check that unit conversion works as intended

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(ls).load() is not None
    assert load_dict(kernel_dict) is not None
