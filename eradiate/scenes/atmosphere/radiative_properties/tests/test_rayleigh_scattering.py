import numpy as np

from eradiate.scenes.atmosphere.radiative_properties.rayleigh import (
    _LOSCHMIDT, kf, sigma_s_single)
from eradiate.util.units import ureg

Q_ = ureg.Quantity


def test_sigma_s_single():
    """Test computation of Rayleigh scattering coefficient with default
    values """

    ref_cross_section = Q_(4.513e-27, "cm**2")
    ref_sigmas = ref_cross_section * _LOSCHMIDT
    expected = ref_sigmas

    # Compare to reference value computed from scattering cross section in
    # Bates (1984) Planetary and Space Science, Volume 32, No. 6.
    assert np.allclose(sigma_s_single(), expected, rtol=1e-2)
