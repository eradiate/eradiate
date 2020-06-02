import numpy as np

from eradiate.scenes.measure import distant
from eradiate.util.units import ureg


def test_distant(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_sensor = distant(45., 0., 32)
    assert load_dict(dict_sensor) is not None

    # With degrees
    assert distant(45. * ureg.deg, 0., 32) == dict_sensor

    # With radian
    assert distant(0.25 * np.pi * ureg.rad, 0., 32) == dict_sensor