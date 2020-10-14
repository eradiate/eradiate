import numpy as np
import pytest

from eradiate.util.units import ureg
from eradiate.scenes.atmosphere.radiative_properties import compute_mono

_Q = ureg.Quantity


def test_compute_monochromatic_radiative_properties():
    from eradiate.scenes.atmosphere.thermophysics.us76 import make_profile
    profile = make_profile()
    with pytest.raises(ValueError):
        compute_mono(
            profile=profile,
            scattering_on=False,
            absorption_on=False
        )

    wavelength = 500.
    properties = compute_mono(
        profile=profile,
        wavelength=wavelength,
        absorption_on=False
    )
    from eradiate.scenes.atmosphere.radiative_properties.rayleigh import sigma_s_air
    sigma_s = sigma_s_air(
        wavelength=wavelength,
        number_density=_Q(profile.n_tot.values, profile.n_tot.units)
    ).to("m^-1")
    assert np.allclose(properties.sigma_s.values, sigma_s.magnitude)
