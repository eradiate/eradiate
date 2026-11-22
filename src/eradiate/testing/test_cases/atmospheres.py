import numpy as np

from ...experiments import AtmosphereExperiment
from ...units import unit_registry as ureg


def create_rpv_afgl1986_continental_brfpp():
    r"""
    *RPV AFGL1986 Aerosol*

    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile
    - Aerosol layer: Uniform layer ranging from 1 km to 2 km, with AOT at 550 nm = 0.5; aerosol dataset ``govaerts_2021-continental``
    - Surface: RPV BSDF with k = 0.95, g = -0.1, and ρ₀ = 0.027685 (vegetated canopy)
    - Illumination: Directional illumination with a zenith angle of 20°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 550 nm
    """

    return AtmosphereExperiment(
        surface={"type": "rpv", "k": 0.95, "g": -0.1, "rho_0": 0.027685},
        illumination={
            "type": "directional",
            "zenith": 20 * ureg.deg,
            "irradiance": 20.0,
        },
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 75.01, 2.0),
            "srf": {"type": "delta", "wavelengths": 550 * ureg.nm},
        },
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "thermoprops": {
                    "identifier": "afgl_1986-us_standard",
                    "z": np.arange(0, 120.1, 0.1) * ureg.km,
                    "additional_molecules": False,
                },
                "absorption_data": "monotropa",
            },
            "particle_components": {
                "type": "particle_ot",
                "bottom": 1 * ureg.km,
                "top": 2 * ureg.km,
                "tau_ref": 0.5,
                "particle_properties": "govaerts_2021-continental",
            },
        },
        integrator={"type": "piecewise_volpath", "moment": True},
    )


def create_rpv_afgl1986_brfpp():
    r"""
    *RPV AFGL1986*

    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile
    - Surface: RPV BSDF with k = 0.95, g = -0.1, and ρ₀ = 0.027685 (vegetated canopy)
    - Illumination: Directional illumination with a zenith angle of 20°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 550 nm
    """
    return AtmosphereExperiment(
        surface={"type": "rpv", "k": 0.95, "g": -0.1, "rho_0": 0.027685},
        illumination={
            "type": "directional",
            "zenith": 20 * ureg.deg,
            "irradiance": 20.0,
        },
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "azimuth": 0.0,
                "zeniths": np.arange(-75.0, 75.01, 2.0),
                "srf": {"type": "delta", "wavelengths": 550 * ureg.nm},
            }
        ],
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "thermoprops": {
                    "identifier": "afgl_1986-us_standard",
                    "z": np.linspace(0, 120, 61) * ureg.km,
                    "additional_molecules": False,
                },
                "absorption_data": "monotropa",
            },
        },
        ckd_quad_config={"ng_max": 16},
        integrator={"type": "piecewise_volpath", "moment": True},
    )
