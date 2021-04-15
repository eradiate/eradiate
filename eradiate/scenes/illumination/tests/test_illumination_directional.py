import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import DirectionalIllumination


def test_directional(mode_mono):
    # Constructor
    d = DirectionalIllumination()
    assert KernelDict.new(d).load() is not None

    # Check if a more detailed spec is valid
    d = DirectionalIllumination(irradiance={"type": "uniform", "value": 1.0})
    assert KernelDict.new(d).load() is not None

    # Check if solar irradiance spectrum can be used
    d = DirectionalIllumination(irradiance={"type": "solar_irradiance"})
    assert KernelDict.new(d).load() is not None

    # Check if specification from a float works
    d = DirectionalIllumination(irradiance=1.0)
    assert KernelDict.new(d).load() is not None

    # Check if specification from a constant works
    d = DirectionalIllumination(irradiance=ureg.Quantity(1.0, "W/m^2/nm"))
    assert KernelDict.new(d).load() is not None
    with pytest.raises(pinttr.exceptions.UnitsError):  # Wrong units
        DirectionalIllumination(irradiance=ureg.Quantity(1.0, "W/m^2/sr/nm"))
