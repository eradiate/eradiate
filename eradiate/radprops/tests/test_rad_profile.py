import numpy as np
import pytest

import eradiate
from eradiate.radprops import (
    ArrayRadProfile, RadProfileFactory, US76ApproxRadProfile
)
from eradiate import unit_registry as ureg


def test_rad_props_profile_factory(mode_mono):
    p = RadProfileFactory.create({
        "type": "array",
        "albedo_values": [[[0, 1, 2, 3]]],
        "sigma_t_values": [[[0, 1, 2, 3]]]
    })
    assert p is not None


def test_array_rad_props_profile(mode_mono):
    p = ArrayRadProfile(albedo_values=np.linspace(0., 1., 11).reshape((1, 1, 11)),
                        sigma_t_values=np.linspace(0., 1e-5, 11).reshape((1, 1, 11)))
    assert np.allclose(p.albedo, ureg.Quantity(np.linspace(0., 1., 11), ureg.dimensionless))
    assert np.allclose(p.sigma_t, ureg.Quantity(np.linspace(0., 1e-5, 11), ureg.m ** -1))
    assert isinstance(p.sigma_a, ureg.Quantity)
    assert isinstance(p.sigma_s, ureg.Quantity)

    # check that to_dataset method does not fail
    assert p.to_dataset()

    with pytest.raises(ValueError):
        p = ArrayRadProfile(albedo_values=np.linspace(0., 1., 11),
                            sigma_t_values=np.linspace(0., 1e-5, 10))


@pytest.mark.slow
def test_us76_approx_rad_profile(mode_mono):
    # We set the wavelength ourselves because the test absorption dataset has
    # a very narrow wavelength range
    eradiate.set_mode("mono", wavelength=ureg.Quantity(630.827266, "nm"))

    # Default constructor
    p = US76ApproxRadProfile()

    for x in [p.sigma_a, p.sigma_s, p.sigma_t, p.albedo]:
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (1, 1, 50)

    # Custom atmosphere height
    p = US76ApproxRadProfile(height=120.)
    for x in [p.sigma_a, p.sigma_s, p.sigma_t, p.albedo]:
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (1, 1, 50)

    # Custom number of layers
    p = US76ApproxRadProfile(n_layers=36)
    for x in [p.sigma_a, p.sigma_s, p.sigma_t, p.albedo]:
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (1, 1, 36)
