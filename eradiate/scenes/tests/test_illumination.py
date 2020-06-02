import numpy as np

from eradiate.scenes.illumination import directional, constant
from eradiate.util.units import ureg


def test_directional(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_emitter = directional(45., 0., 10.)
    assert load_dict(dict_emitter) is not None

    # With degrees
    assert directional(45. * ureg.deg, 0., 10.) == dict_emitter

    # With radian
    assert directional(0.25 * np.pi * ureg.rad, 0., 10.) == dict_emitter


def test_constant(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_emitter = constant(10.)
    assert load_dict(dict_emitter) is not None