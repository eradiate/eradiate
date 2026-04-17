import numpy as np

from ..util import append_doc
from ...experiments import AtmosphereExperiment, CanopyAtmosphereExperiment
from ...units import unit_registry as ureg


def _create_integrator_surface(
    absorption_database_error_handler_config, integrator_dict
):
    r"""
    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile
    - Aerosol layer: Uniform layer ranging from 1 km to 2 km, with AOT at 550 nm = 0.2; aerosol dataset ``govaerts_2021-continental``
    - Surface: RPV BSDF with k = 0.95, g = -0.1, and ρ₀ = 0.027685 (vegetated canopy)
    - Illumination: Directional illumination with a zenith angle of 20°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 550 nm
    - Integrator: Integrator passed in `integrator_dict`.
    """

    return AtmosphereExperiment(
        surface={"type": "rpv", "k": 0.95, "g": -0.1, "rho_0": 0.027685},
        illumination={
            "type": "directional",
            "zenith": 20 * ureg.deg,
            "irradiance": 2.0,
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
                "error_handler_config": absorption_database_error_handler_config,
            },
            "particle_layers": {
                "type": "particle_layer",
                "bottom": 1 * ureg.km,
                "top": 2 * ureg.km,
                "tau_ref": 0.2,
                "dataset": "govaerts_2021-continental",
            },
        },
        integrator=integrator_dict,
    )


def create_integrator_canopy(absorption_database_error_handler_config, integrator_dict):
    r"""
    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile
    - Aerosol layer: Uniform layer ranging from 0 km to 2 km, with AOT at 550 nm = 0.2; aerosol dataset ``govaerts_2021-continental``
    - Canopy: Homogeneous discrete canopy with LAI = 3.0, leaf radius = 0.05 m, horizontal extent = 5 m, vertical extent = 2 m
    - Surface: Lambertian BSDF with reflectance = 0.5
    - Illumination: Directional illumination with a zenith angle of 30°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 550 nm
    - Integrator: Integrator passed in `integrator_dict`.
    """

    return CanopyAtmosphereExperiment(
        surface={
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.5},
        },
        illumination={
            "type": "directional",
            "zenith": 30 * ureg.deg,
            "azimuth": 0.0 * ureg.deg,
        },
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "azimuth": 0.0 * ureg.deg,
                "zeniths": np.arange(-75.0, 75.01, 2.0) * ureg.deg,
                "srf": {"type": "delta", "wavelengths": 550 * ureg.nm},
                "target": {
                    "type": "rectangle",
                    "xmin": -2.5 * ureg.m,
                    "xmax": 2.5 * ureg.m,
                    "ymin": -2.5 * ureg.m,
                    "ymax": 2.5 * ureg.m,
                    "z": 2.0 * ureg.m,
                },
            }
        ],
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
                "bottom": 0 * ureg.km,
                "top": 2 * ureg.km,
                "distribution": {"type": "uniform"},
                "tau_ref": 0.2,
                "dataset": "govaerts_2021-continental",
            },
        },
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.05 * ureg.m,
            "l_horizontal": 5.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
            "nu": 1.0,
            "mu": 1.0,
            "leaf_reflectance": 0.05653,
            "leaf_transmittance": 0.01692,
            "padding": 5,
        },
        integrator=integrator_dict,
    )


@append_doc(_create_integrator_surface, prepend=True)
def create_eovolpath_surface(absorption_database_error_handler_config):
    r"""
    *EOVolpath Integrator*

    """

    integrator = {"type": "eovolpath", "ddis_threshold": 0.1, "moment": True}

    return _create_integrator_surface(
        absorption_database_error_handler_config, integrator
    )


@append_doc(_create_integrator_surface, prepend=True)
def create_volpath_surface(absorption_database_error_handler_config):
    r"""
    *Volpath Integrator*

    """

    integrator = {"type": "volpath", "moment": True}

    return _create_integrator_surface(
        absorption_database_error_handler_config, integrator
    )


@append_doc(create_integrator_canopy, prepend=True)
def create_eovolpath_canopy(absorption_database_error_handler_config):
    r"""
    *EOVolpath Integrator*

    """

    integrator = {"type": "eovolpath", "ddis_threshold": 0.1, "moment": True}

    return create_integrator_canopy(
        absorption_database_error_handler_config, integrator
    )


@append_doc(create_integrator_canopy, prepend=True)
def create_volpath_canopy(absorption_database_error_handler_config):
    r"""
    *Volpath Integrator*

    """

    integrator = {"type": "volpath", "moment": True}

    return create_integrator_canopy(
        absorption_database_error_handler_config, integrator
    )
