import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import unit_registry as ureg


@pytest.fixture(scope="module")
def onedim_rayleigh_radprops():
    """
    Radiative properties of a Rayleigh scattering medium in [280, 2400] nm.

    The Rayleigh scattering phase function is computed from the analytical
    formula and tabulated on a regular scattering angle cosine grid.
    The Rayleigh scattering phase function does not depend on wavelength.

    The Rayleigh scattering coefficient is computed with
    :meth:`~eradiate.radprops.rayleigh.compute_sigma_s_air` with default
    the number density, i.e., standard air number density.

    The single-scattering albedo can be set but it is assumed to be spectrally
    invariant.

    This fixture returns a callable that takes the single-scattering albedo as
    argument and returns the corresponding Rayleigh radiative properties data
    set.
    """

    def radprops(albedo: float = 1.0) -> xr.DataArray:
        w = np.linspace(279.0, 2401.0, 10000) * ureg.nm

        # Collision coefficients
        sigma_s = eradiate.radprops.rayleigh.compute_sigma_s_air(wavelength=w)
        albedo = albedo * np.ones_like(sigma_s) * ureg.dimensionless
        sigma_t = (
            np.divide(
                sigma_s.m,
                albedo.m,
                where=albedo.m != 0,
                out=sigma_s.m,
            )
            * sigma_s.units
        )

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
                        standard_name="single_scattering_albedo",
                        units=f"{albedo.units:~}",
                    ),
                ),
                "phase": (
                    ("w", "mu", "i", "j"),
                    phase.magnitude,
                    dict(
                        standard_name="scattering_phase_matrix",
                        units=f"{phase.units:~}",
                    ),
                ),
            },
            coords={
                "w": ("w", w.magnitude, dict(units=f"{w.units:~}")),
                "mu": (
                    "mu",
                    mu,
                    dict(
                        standard_name="scattering_angle_cosine", units="dimensionless"
                    ),
                ),
                "i": ("i", [0]),
                "j": ("j", [0]),
            },
        )

    return radprops
