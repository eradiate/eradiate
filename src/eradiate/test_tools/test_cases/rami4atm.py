import numpy as np

from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment


def create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp():
    r"""
    *RAMI4ATM HOM00_BLA_SD2S_M03*

    This scenario is based on the ``HOM00_BLA_SD2S_M03_z30a000-brfpp`` scenario
    of the RAMI4ATM benchmark.

    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile
    - Aerosol layer: Uniform layer ranging from 0 km to 2 km, with AOT at 550 nm = 0.5; aerosol dataset ``govaerts_2021-desert``
    - Surface: Black
    - Illumination: Directional illumination with a zenith angle of 30°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 550 nm
    """
    config = {
        "surface": {
            "reflectance": {"value": 0.0, "type": "uniform"},
            "type": "lambertian",
        },
        "atmosphere": {
            "molecular_atmosphere": {
                "has_absorption": False,
                "has_scattering": True,
                "type": "molecular",
                "thermoprops": {
                    "identifier": "afgl_1986-us_standard",
                    "z": np.arange(0, 120.05, 0.05) * ureg.km,
                },
                "absorption_data": "monotropa",
            },
            "particle_layers": [
                {
                    "bottom": 0,
                    "bottom_units": "meter",
                    "top": 2000,
                    "top_units": "meter",
                    "distribution": {"type": "uniform"},
                    "tau_ref": 0.2,
                    "dataset": "govaerts_2021-desert",
                }
            ],
            "type": "heterogeneous",
        },
        "illumination": {
            "zenith": 30.0,
            "zenith_units": "degree",
            "azimuth": 0.0,
            "azimuth_units": "degree",
            "type": "directional",
        },
        "measures": [
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 2),
                "zeniths_units": "degree",
                "azimuth": 0.0,
                "azimuth_units": "degree",
                "srf": "sentinel_2a-msi-3",
                "spp": 1000,
            }
        ],
        "ckd_quad_config": {
            "policy": "fixed",
            "type": "gauss_legendre",
            "ng_max": 16,
        },
        "integrator": {"type": "piecewise_volpath", "moment": True},
    }

    return AtmosphereExperiment(**config)
