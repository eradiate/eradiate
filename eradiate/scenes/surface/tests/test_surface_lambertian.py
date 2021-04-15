from eradiate.scenes.core import KernelDict
from eradiate.scenes.surface import LambertianSurface


def test_lambertian(mode_mono):
    # Default constructor
    ls = LambertianSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.new(ls)
    assert kernel_dict.load() is not None

    # Constructor with arguments
    ls = LambertianSurface(width=1000.0, reflectance={"type": "uniform", "value": 0.3})

    # Check if produced scene can be instantiated
    assert KernelDict.new(ls).load() is not None
