import numpy as np

from eradiate.scenes import SceneDict
from eradiate.scenes.measure import _distant, Distant
from eradiate.util.units import ureg


def test_distant_function(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_sensor = _distant(45., 0., 32)
    assert load_dict(dict_sensor) is not None

    # With degrees
    assert _distant(45. * ureg.deg, 0., 32) == dict_sensor

    # With radian
    assert _distant(0.25 * np.pi * ureg.rad, 0., 32) == dict_sensor


def test_distant_class(variant_scalar_mono):
    # Constructor
    d = Distant()
    assert d.kernel_dict()[d.id] == _distant()
    assert SceneDict.empty().add(d) is not None
