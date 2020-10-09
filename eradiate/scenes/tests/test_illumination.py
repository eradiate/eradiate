from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import ConstantIllumination, DirectionalIllumination


def test_directional_class(mode_mono):
    # Constructor
    d = DirectionalIllumination()
    assert KernelDict.empty().add(d).load() is not None

    # Check if a more detailed spec is valid
    d = DirectionalIllumination(irradiance={"type": "uniform", "value": 1.0})
    assert KernelDict.empty().add(d).load() is not None


def test_constant(mode_mono):
    # Constructor
    c = ConstantIllumination()
    assert c.kernel_dict()[c.id] == {"type": "constant", "radiance": {"type": "uniform", "value": 1.0}}
    assert KernelDict.empty().add(c).load() is not None

    # Check if a more detailed spec is valid
    d = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert KernelDict.empty().add(d).load() is not None
