import pinttr
import pytest

from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import ConstantIllumination, DirectionalIllumination
from eradiate import unit_registry as ureg


def test_constant(mode_mono):
    # Constructor
    c = ConstantIllumination()
    assert c.kernel_dict()[c.id] == {
        "type": "constant",
        "radiance": {"type": "uniform", "value": 1.0}
    }
    assert KernelDict.empty().add(c).load() is not None

    # Check if a more detailed spec is valid
    c = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert KernelDict.empty().add(c).load() is not None

    # Check if 'uniform' shortcut works
    c = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert KernelDict.empty().add(c).load() is not None

    # Check if super lazy way works too
    c = ConstantIllumination(radiance=1.0)
    assert KernelDict.empty().add(c).load() is not None


def test_directional(mode_mono):
    # Constructor
    d = DirectionalIllumination()
    assert KernelDict.empty().add(d).load() is not None

    # Check if a more detailed spec is valid
    d = DirectionalIllumination(irradiance={"type": "uniform", "value": 1.0})
    assert KernelDict.empty().add(d).load() is not None

    # Check if solar irradiance spectrum can be used
    d = DirectionalIllumination(irradiance={"type": "solar_irradiance"})
    assert KernelDict.empty().add(d).load() is not None

    # Check if specification from a float works
    d = DirectionalIllumination(irradiance=1.)
    assert KernelDict.empty().add(d).load() is not None

    # Check if specification from a constant works
    d = DirectionalIllumination(irradiance=ureg.Quantity(1., "W/m^2/nm"))
    assert KernelDict.empty().add(d).load() is not None
    with pytest.raises(pinttr.exceptions.UnitsError):  # Wrong units
        DirectionalIllumination(irradiance=ureg.Quantity(1., "W/m^2/sr/nm"))
