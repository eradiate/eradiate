import numpy as np

from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import _directional, DirectionalIllumination, _constant, ConstantIllumination
from eradiate.util.units import ureg


def test_directional_function(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_emitter = _directional(45., 0., 10.)
    assert load_dict(dict_emitter) is not None

    # With degrees
    assert _directional(45. * ureg.deg, 0., 10.) == dict_emitter

    # With radian
    assert _directional(0.25 * np.pi * ureg.rad, 0., 10.) == dict_emitter


def test_directional_class(variant_scalar_mono):
    # Constructor
    d = DirectionalIllumination()
    assert d.kernel_dict()[d.id] == _directional()
    assert KernelDict.empty().add(d).load() is not None


def test_constant_function(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_emitter = _constant(10.)
    assert load_dict(dict_emitter) is not None


def test_constant_class(variant_scalar_mono):
    # Constructor
    c = ConstantIllumination()
    assert c.kernel_dict()[c.id] == _constant()
    assert KernelDict.empty().add(c).load() is not None
