import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import ConstantIllumination, DirectionalIllumination


def test_constant(mode_mono):
    # Constructor
    c = ConstantIllumination()
    assert c.kernel_dict()[c.id] == {
        "type": "constant",
        "radiance": {"type": "uniform", "value": 1.0},
    }
    assert KernelDict.new(c).load() is not None

    # Check if a more detailed spec is valid
    c = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert KernelDict.new(c).load() is not None

    # Check if 'uniform' shortcut works
    c = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert KernelDict.new(c).load() is not None

    # Check if super lazy way works too
    c = ConstantIllumination(radiance=1.0)
    assert KernelDict.new(c).load() is not None
