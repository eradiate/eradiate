from eradiate.scenes.core import KernelDict
from eradiate.scenes.lithosphere import LambertianSurface, RPVSurface


def test_lambertian(mode_mono):
    # Default constructor
    ls = LambertianSurface()
    assert ls.config == {
        "id": "surface",
        "reflectance": {"type": "uniform", "value": .5, "quantity": None},
        "width": 1.,
        "width_unit": "meter",
    }

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = LambertianSurface.from_dict({
        "width": 1000.,
        "reflectance": {"type": "uniform", "value": .3}
    })

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
    ls = RPVSurface.from_dict({"width": 1000., "rho_0": 0.3, "k": 1.4, "ttheta": -0.23})

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(ls).load() is not None
    assert load_dict(kernel_dict) is not None
