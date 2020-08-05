import numpy as np

from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import ConstantIllumination, DirectionalIllumination, _directional
from eradiate.util.units import ureg


def test_directional_function(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_emitter = _directional(45., 0., 10.)
    assert load_dict(dict_emitter) is not None

    # With degrees
    assert _directional(45. * ureg.deg, 0., 10.) == dict_emitter

    # With radian
    assert _directional(0.25 * np.pi * ureg.rad, 0., 10.) == dict_emitter


def test_directional_class(mode_mono):
    # Constructor
    d = DirectionalIllumination()
    assert KernelDict.empty().add(d).load() is not None

    # Check if a more detailed spec is valid
    d = DirectionalIllumination({
        "irradiance": {"type": "uniform", "value": 1.0}
    })
    assert d.kernel_dict()[d.id] == _directional()
    assert KernelDict.empty().add(d).load() is not None


def test_constant(mode_mono):
    # Constructor
    c = ConstantIllumination()
    assert c.kernel_dict()[c.id] == {"type": "constant", "radiance": {"type": "uniform", "value": 1.0}}
    assert KernelDict.empty().add(c).load() is not None

    # Check if a more detailed spec is valid
    d = ConstantIllumination({
        "radiance": {"type": "uniform", "value": 1.0}
    })
    assert KernelDict.empty().add(d).load() is not None
