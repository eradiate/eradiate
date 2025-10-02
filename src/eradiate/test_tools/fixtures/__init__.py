"""
Eradiate-specific pytest fixtures, imported automatically upon starting a pytest
session (see ``conftest.py``).
"""

from functools import partial

import joseki
import numpy as np
import pytest

from .data import make_particle_dataset
from .util import check_plugin
from .. import fresolver
from .. import unit_registry as ureg


@pytest.fixture
def plugin_checker(request):
    """
    Fixture to check if a pytest plugin is loaded and enabled.

    Examples
    --------
    In a pytest case:

    >>> def test_something(plugin_checker):
    ...     if plugin_checker("robotframework"):
    ...         print("robotframework plugin is loaded and enabled.")
    """
    return partial(check_plugin, request.config)


@pytest.fixture
def has_robot(plugin_checker):
    """
    Fixture that returns ``True`` iff the robotframework plugin is loaded and
    enabled in the current session.
    """
    return plugin_checker("robotframework")


@pytest.fixture
def absorption_database_error_handler_config():
    """
    Error handler configuration for absorption coefficient interpolation.

    Notes
    -----
    This configuration is chosen to ignore all interpolation issues (except
    bounds error along the mole fraction dimension) because warnings are
    captured by pytest which will raise.
    Ignoring the bounds on pressure and temperature is safe because
    out-of-bounds values usually correspond to locations in the atmosphere
    that are so high that the contribution to the absorption coefficient
    are negligible at these heights.
    The bounds error for the 'x' (mole fraction) coordinate is considered
    fatal.
    """
    return {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    }


@pytest.fixture
def atmosphere_us_standard_ckd(
    thermoprops_us_standard, absorption_database_error_handler_config
):
    return {
        "type": "molecular",
        "thermoprops": thermoprops_us_standard,
        "absorption_data": "monotropa",
        "error_handler_config": absorption_database_error_handler_config,
    }


@pytest.fixture
def atmosphere_us_standard_mono(
    thermoprops_us_standard, absorption_database_error_handler_config
):
    """
    AFGL (1986) U.S. Standard atmosphere with monochromatic absorption data.

    Notes
    -----
    Molecules included are H2O, CO2, O3, N2O, CO, CH4, O2.
    Specified absorption data covers the wavelength range [250, 3125] nm.
    Altitude grid is regular with a 1 km step, from 0 to 120 km.
    """
    return {
        "type": "molecular",
        "thermoprops": thermoprops_us_standard,
        "absorption_data": "komodo",
        "error_handler_config": absorption_database_error_handler_config,
    }


@pytest.fixture
def atmosphere_cams_lybia4_ckd(absorption_database_error_handler_config):
    """
    CAMS Lybia4 atmosphere with CKD absorption data.

    Notes
    -----
    Molecules included are H2O, CO2, O3, N2O, CO, CH4, O2, NO2, NO, SO2.
    Specified absorption data covers the CKD band associated with wavenumber
    interval [18100, 18200] cm^-1, i.e. the wavelenght range
    [549.45, 552.48] nm.
    Altitude grid is regular with a 1 km step, from 0 to 120 km.
    """
    thermoprops = fresolver.load_dataset("tests/thermoprops/cams_lybia4_2005-04-01.nc")

    return {
        "type": "molecular",
        "thermoprops": thermoprops,
        "absorption_data": "monotropa",
        "error_handler_config": absorption_database_error_handler_config,
    }


@pytest.fixture
def ert_seed_state():
    from eradiate.rng import SeedState

    return SeedState(0)


@pytest.fixture
def particle_dataset_absorbing_only():
    """Absorbing only particles radiative properties data set fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = np.zeros_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    return make_particle_dataset(
        albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w
    )


@pytest.fixture
def particle_dataset_scattering_only():
    """Scattering only particles radiative properties data set fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = np.ones_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    return make_particle_dataset(
        albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w
    )


@pytest.fixture
def particle_dataset_test():
    """
    Particles radiative properties data set fixture.

    Particle radiative properties are spectrally constant with values specified
    below:
        * the phase function is isotropic
        * the extinction coefficient is 1 / km
        * the albedo is 0.8
    """
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = 0.8 * np.ones_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    return make_particle_dataset(
        albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w
    )


@pytest.fixture(scope="session")
def session_timestamp():
    from datetime import datetime

    return datetime.now()


@pytest.fixture(scope="session")
def thermoprops_us_standard():
    yield joseki.make(
        identifier="afgl_1986-us_standard",
        z=np.linspace(0.0, 120.0, 121) * ureg.km,
        additional_molecules=False,
    )
