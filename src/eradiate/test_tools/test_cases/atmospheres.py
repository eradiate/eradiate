import numpy as np

from ...experiments import AtmosphereExperiment
from ...units import unit_registry as ureg


# TODO : This probably needs to go somewhere else or get called by fixtures
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


def create_rpv_afgl1986_continental_brfpp(absorption_database_error_handler_config):
    r"""
    RPV AFGL1986 Aerosol
    ====================

    Rationale
    ---------

    This test case uses a basic atmospheric scene:

    * RPV surface emulating a canopy
    * Molecular atmosphere following the AFGL 1986 model
    * Aerosol layer at 1km elevation

    Parameters

    * Atmosphere: Molecular atmosphere using the agfl1986 profile
    * Aerosol layer: 16 layers from 1km to 2km height, :math:`\tau_{500} = 0.5`
      Radiative properties from the Govaerts 2021 Continental dataset
    * Surface: Square surface with RPV BSDF with :math:`k = 0.95`, :math:`g = -0.1`
      and :math:`\rho_0 = 0.027685`
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Distant reflectance measure, covering a plane, (76 angular points)
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
            "srf": {"type": "multi_delta", "wavelengths": 550 * ureg.nm},
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
                "error_handler_config": absorption_database_error_handler_config,
            },
            "particle_layers": {
                "type": "particle_layer",
                "bottom": 1 * ureg.km,
                "top": 2 * ureg.km,
                "tau_ref": 0.5,
                "dataset": "govaerts_2021-continental",
            },
        },
    )


def create_rpv_afgl1986_brfpp(absorption_database_error_handler_config):
    r"""
    RPV AFGL1986
    ============

    Rationale
    ---------

    This test case uses a basic atmospheric scene:

    * RPV surface emulating a canopy
    * Molecular atmosphere following the AFGL 1986 model

    Parameters

    * Atmosphere: Molecular atmosphere using the agfl1986 profile
    * Surface: Square surface with RPV BSDF with :math:`k = 0.95`, :math:`g = -0.1`
      and :math:`\rho_0 = 0.027685`
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Distant reflectance measure, covering a plane, (76 angular points)
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
                "srf": {"type": "multi_delta", "wavelengths": 550 * ureg.nm},
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
                "error_handler_config": absorption_database_error_handler_config,
            },
        },
    )
