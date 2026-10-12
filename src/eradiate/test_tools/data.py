import xarray as xr

from ..units import symbol


def make_particle_dataset(albedo, sigma_t, phase, mu, w):
    return xr.Dataset(
        data_vars={
            "sigma_t": (
                "w",
                sigma_t.magnitude,
                {
                    "standard_name": "air_volume_extinction_coefficient",
                    "units": symbol(sigma_t.units),
                },
            ),
            "albedo": (
                "w",
                albedo.magnitude,
                {
                    "standard_name": "single_scattering_albedo",
                    "units": symbol(albedo.units),
                },
            ),
            "phase": (
                ("w", "mu", "i", "j"),
                phase.magnitude,
                {
                    "standard_name": "scattering_phase_matrix",
                    "units": symbol(phase.units),
                },
            ),
        },
        coords={
            "w": ("w", w.magnitude, {"units": symbol(w.units)}),
            "mu": (
                "mu",
                mu.magnitude,
                {
                    "standard_name": "scattering_angle_cosine",
                    "units": symbol(mu.units),
                },
            ),
            "i": ("i", [0]),
            "j": ("j", [0]),
        },
    )
