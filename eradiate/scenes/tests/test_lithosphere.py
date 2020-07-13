from eradiate.scenes.core import KernelDict
from eradiate.scenes.lithosphere import LambertianSurface, RPVSurface


def test_lambertian(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = LambertianSurface()
    assert ls.config == {"reflectance": .5, "width": 1.}

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = LambertianSurface.from_dict({"width": 1000., "reflectance": 0.3})

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(ls).load() is not None


def test_rpv(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = RPVSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = RPVSurface.from_dict({"width": 1000., "rho_0": 0.3, "k": 1.4, "ttheta": -0.23})

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(ls).load() is not None
    assert load_dict(kernel_dict) is not None

