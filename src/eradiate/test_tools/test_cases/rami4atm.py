import numpy as np

from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment


def create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp():
    r"""
    RAMI4ATM HOM00_BLA_SD2S_M03
    ===========================

    Rationale
    ---------
    This test case uses a basic atmospheric scene:

    * black surface
    * Molecular atmosphere following the AFGL 1986 model
    * Aerosol layer at 1km elevation

    Parameters

    * Atmosphere: molecular atmosphere using the AFGL 1986 profile
      (U.S. standard)
    * Aerosol layer: uniform, covers [0, 2] km extent, :math:`\tau_{500} = 0.2`;
      radiative properties from the ``govaerts_2021-desert`` dataset
    * Surface: black
    * Illumination: directional illumination with a zenith angle
      :math:`\theta = 30°`
    * Measure: distant measure, covering the principal plane for
      :math:`\theta \in [-75, 75]°`
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
        "quad_spec": {
            "type": "fixed",
            "quad_type": "gauss_legendre",
            "n": 16,
        },
    }

    return AtmosphereExperiment(**config)
    
