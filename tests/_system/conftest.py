import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg


@pytest.fixture(scope="module")
def onedim_rayleigh_radprops():
    """
    Radiative properties of a Rayleigh scattering medium in [280, 2400] nm.
    """
    w = np.linspace(279.0, 2401.0, 10000) * ureg.nm

    # Collision coefficients
    sigma_s = eradiate.radprops.rayleigh.compute_sigma_s_air(wavelength=w)
    albedo = np.ones_like(sigma_s) * ureg.dimensionless  # no absorption
    sigma_t = sigma_s * albedo

    # Phase function
    # Note: rayleigh phase function does not change with wavelength
    def rayleigh_phase_function(mu):
        magnitude = 3.0 * (1 + np.square(mu)) / (16 * np.pi)
        return magnitude / ureg.steradian

    mu = np.linspace(-1.0, 1.0)
    arrays = [rayleigh_phase_function(mu) for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)

    return xr.Dataset(
        data_vars={
            "sigma_t": (
                "w",
                sigma_t.magnitude,
                dict(
                    standard_name="air_volume_extinction_coefficient",
                    units=f"{sigma_t.units:~}",
                ),
            ),
            "albedo": (
                "w",
                albedo.magnitude,
                dict(
                    standard_name="single_scattering_albedo", units=f"{albedo.units:~}"
                ),
            ),
            "phase": (
                ("w", "mu", "i", "j"),
                phase.magnitude,
                dict(standard_name="scattering_phase_matrix", units=f"{phase.units:~}"),
            ),
        },
        coords={
            "w": ("w", w.magnitude, dict(units=f"{w.units:~}")),
            "mu": (
                "mu",
                mu,
                dict(standard_name="scattering_angle_cosine", units="dimensionless"),
            ),
            "i": ("i", [0]),
            "j": ("j", [0]),
        },
    )
