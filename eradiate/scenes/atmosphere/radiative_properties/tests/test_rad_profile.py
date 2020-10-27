import numpy as np
import pytest

from eradiate.scenes.atmosphere.radiative_properties.rad_profile \
    import ArrayRadProfile, RadProfileFactory
from eradiate.util.units import ureg


def test_rad_props_profile_factory(mode_mono):
    p = RadProfileFactory.create({
        "type": "array",
        "albedo_values": [0, 1, 2, 3],
        "sigma_t_values": [0, 1, 2, 3]
    })
    assert p is not None


def test_array_rad_props_profile(mode_mono):
    p = ArrayRadProfile(albedo_values=np.linspace(0., 1., 11),
                        sigma_t_values=np.linspace(0., 1e-5, 11))
    assert np.allclose(p.albedo, ureg.Quantity(np.linspace(0., 1., 11), ureg.dimensionless))
    assert np.allclose(p.sigma_t, ureg.Quantity(np.linspace(0., 1e-5, 11), ureg.m ** -1))
    assert isinstance(p.sigma_a, ureg.Quantity)
    assert isinstance(p.sigma_s, ureg.Quantity)

    with pytest.raises(ValueError):
        p = ArrayRadProfile(albedo_values=np.linspace(0., 1., 11),
                            sigma_t_values=np.linspace(0., 1e-5, 10))
