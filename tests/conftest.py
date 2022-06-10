import os

import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg
from eradiate.units import symbol

eradiate.plot.set_style()

# ------------------------------------------------------------------------------
#               Customizable output dir for test artifacts
# ------------------------------------------------------------------------------


def pytest_addoption(parser):
    eradiate_source_dir = os.environ.get("ERADIATE_SOURCE_DIR", ".")
    parser.addoption(
        "--artefact-dir",
        action="store",
        default=os.path.join(eradiate_source_dir, "build/test_artefacts/"),
    )


# See: https://stackoverflow.com/a/55301318/3645374
@pytest.fixture(scope="session")
def artefact_dir(pytestconfig):
    option_value = pytestconfig.getoption("artefact_dir")

    if not os.path.isdir(option_value):
        os.makedirs(option_value)

    return option_value


# ------------------------------------------------------------------------------
#                              Other configuration
# ------------------------------------------------------------------------------


def pytest_configure(config):
    markexpr = config.getoption("markexpr", "False")
    has_slow = "not slow" not in markexpr
    has_regression = "not regression" not in markexpr

    if has_slow:
        print(
            "\033[93m"
            "Running slow tests. To skip them, please run "
            "'pytest -m \"not slow\"' "
            "\033[0m"
        )

    if has_regression:
        print(
            "\033[93m"
            "Running regression tests. To skip them, please run "
            "'pytest -m \"not regression\"' "
            "\033[0m"
        )

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )

    config.addinivalue_line(
        "markers",
        "regression: marks tests as potentially very slow regression tests "
        "(deselect with -m 'not regression')",
    )


@pytest.fixture(scope="session")
def session_timestamp():
    from datetime import datetime

    return datetime.now()


@pytest.fixture
def absorbing_only():
    """Absorbing only particles radiative properties data set fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = np.zeros_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    return to_dataset(albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w)


@pytest.fixture
def scattering_only():
    """Scattering only particles radiative properties data set fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = np.ones_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    return to_dataset(albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w)


@pytest.fixture
def test_particles_dataset():
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
    return to_dataset(albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w)


def to_dataset(albedo, sigma_t, phase, mu, w):
    return xr.Dataset(
        data_vars={
            "sigma_t": (
                "w",
                sigma_t.magnitude,
                dict(
                    standard_name="air_volume_extinction_coefficient",
                    units=symbol(sigma_t.units),
                ),
            ),
            "albedo": (
                "w",
                albedo.magnitude,
                dict(
                    standard_name="single_scattering_albedo",
                    units=symbol(albedo.units),
                ),
            ),
            "phase": (
                ("w", "mu", "i", "j"),
                phase.magnitude,
                dict(
                    standard_name="scattering_phase_matrix",
                    units=symbol(phase.units),
                ),
            ),
        },
        coords={
            "w": ("w", w.magnitude, dict(units=symbol(w.units))),
            "mu": (
                "mu",
                mu.magnitude,
                dict(
                    standard_name="scattering_angle_cosine",
                    units=symbol(mu.units),
                ),
            ),
            "i": ("i", [0]),
            "j": ("j", [0]),
        },
    )
