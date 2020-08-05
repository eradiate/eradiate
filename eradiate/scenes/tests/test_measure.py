import numpy as np

from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import _distant, DistantMeasure, _perspective, PerspectiveCameraMeasure
from eradiate.util.units import ureg


def test_distant_function(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_sensor = _distant(45., 0., 32)
    assert load_dict(dict_sensor) is not None

    # With degrees
    assert _distant(45. * ureg.deg, 0., 32) == dict_sensor

    # With radian
    assert _distant(0.25 * np.pi * ureg.rad, 0., 32) == dict_sensor


def test_distant_class(mode_mono):
    # Constructor
    d = DistantMeasure()
    assert d.kernel_dict()[d.id] == _distant()


def test_perspective_function(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_sensor = _perspective([0, 0, 0], 45., 180., 10., 32, 64)
    assert load_dict(dict_sensor) is not None

    # With degrees
    assert _perspective([0, 0, 0], 45. * ureg.deg, 180., 10., 32, 64) == dict_sensor

    # With radian
    assert _perspective([0, 0, 0], 0.25 * np.pi * ureg.rad, 180., 10., 32, 64) == dict_sensor

    # With kilometers
    assert _perspective([0, 0, 0] * ureg.km, 45., 180., 10. * ureg.km, 32, 64) == dict_sensor

    # With meters
    assert _perspective([0, 0, 0] * ureg.m, 45., 180., 10000. * ureg.m, 32, 64) == dict_sensor


def test_perspective_class(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    assert d.kernel_dict()[d.id] == _perspective()
    assert KernelDict.empty().add(d) is not None
