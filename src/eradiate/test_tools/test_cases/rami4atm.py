import functools

import numpy as np

from ...experiments import CanopyAtmosphereExperiment
from ...units import unit_registry as ureg


def create_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp(spp=1000):
    r"""
    *RAMI4ATM HOM00_BLA_S00S_M04*

    This scenario is based on the ``HOM00_BLA_S00S_M04_z30a000-brfpp`` scenario
    of the RAMI4ATM benchmark.

    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile, absorption only
    - Surface: Black
    - Illumination: Directional illumination with a zenith angle of 30°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 660 nm
    """
    # TODO: This test case definition is kept for compatibility
    return create_rami4atm("hom00_bla_a00s_m04_z30a000_brfpp", spp)


def create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp(spp=1000):
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
    # TODO: This test case definition is kept for compatibility
    return create_rami4atm("hom00_bla_sd2s_m03_z30a000_brfpp", spp)


def create_rami4atm(id: str, spp: int):
    """
    Create an :class:`.Experiment` to simulate one of the RAMI4ATM benchmarking
    cases.

    Parameters
    ----------
    id : str
        Identifier of the test case (see notes for supported values).

    spp : int
        Default measure sample count.

    Returns
    -------
    .Experiment

    Raises
    ------
    ValueError
        If the passed ID contains unhandled parameter values.

    Notes
    -----
    Case IDs are structured as follows (see also the
    `RAMI4ATM case naming convention reference\
    <https://rami-benchmark.jrc.ec.europa.eu/_www/format_RAMI4ATM.php?strPhase=RAMI4ATM>`__::

    <canopy_id>_<surface_id>_<band_id>_<illumination_id>_<measure_id>

    .. list-table::
        :widths: auto

        * - Parameter
          - Accepted values
        * - ``canopy_id``
          - ``hom00`` (none), ``hom45`` (uniform cloud leaf)
        * - ``surface_id``
          - ``bla`` (black), ``whi`` (white), ``lam`` (Lambertian),
            ``rpv`` (RPV), ``rli`` (Ross-Li)
        * - ``atmosphere_id``
          - ``MATP`` with

            * ``M`` in [``0`` (no molecular component),
              ``a`` (molecular absorption only),
              ``s`` (molecular scattering only) or
              ``e`` (molecular absorption and scattering)]
            * ``A`` in [``0`` (no aerosols), ``d`` (desert aerosols) or
              ``c`` (continental aerosols)]
            * ``T`` in [``0`` (no aerosols), ``2`` (AOT = 0.2) or
              ``6`` (AOT = 0.6)]
            * ``P`` in [``s`` (U.S. Standard atmospheric profile)]
        * - ``band_id``
          - ``m04`` (Sentinel-2 / MSI, band 4)
        * - ``illumination_id``
          - ``z30a000`` (30° zenith, 0° azimuth)
        * - ``measure_id``
          - ``brfpp`` (TOA BRF in the principal plane)
    """
    canopy_id, surf_id, atm_id, band_id, illumination_id, measure_id = id.split("_")

    # Canopy setup
    if canopy_id == "hom00":
        canopy = None
    elif canopy_id == "hom45":
        canopy = {
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
            "padding": 20,
        }
    else:
        raise ValueError(f"Unhandled canopy id '{canopy_id}'")

    # Atmosphere setup
    if surf_id == "bla":
        surface = {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.0},
        }
    elif surf_id == "whi":
        surface = {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 1.0},
        }
    elif surf_id == "lam":
        surface = {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.02806},
        }
    elif surf_id == "rpv":
        surface = {
            "type": "rpv",
            "rho_0": {"type": "uniform", "value": 0.017051},
            "k": {"type": "uniform", "value": 0.95},
            "g": {"type": "uniform", "value": -0.1},
            "rho_c": {"type": "uniform", "value": 0.017051},
        }
    elif surf_id == "rli":
        surface = {
            "type": "rtls",
            "f_iso": {"type": "uniform", "value": 0.032171},
            "f_vol": {"type": "uniform", "value": -0.002886},
            "f_geo": {"type": "uniform", "value": 0.001949},
        }
    else:
        raise ValueError(f"Unhandled surface id '{surf_id}'")

    # Atmosphere setup
    component_id, aerosol_id, aerosol_ot = atm_id[:3]

    if component_id == "0":
        molecular_atmosphere = None

    elif component_id in {"a", "s", "e"}:
        molecular_atmosphere = {
            "type": "molecular",
            "thermoprops": {
                "identifier": "afgl_1986-us_standard",
                "z": np.arange(0, 120.05, 0.05) * ureg.km,
            },
            "absorption_data": "monotropa",
        }

        if component_id == "a":
            molecular_atmosphere.update(
                {"has_absorption": True, "has_scattering": False}
            )
        elif component_id == "s":
            molecular_atmosphere.update(
                {"has_absorption": False, "has_scattering": True}
            )
        else:  # component_id == "e":
            molecular_atmosphere.update(
                {"has_absorption": True, "has_scattering": True}
            )

    else:
        raise ValueError(
            f"Unhandled molecular atmosphere component id '{component_id}'"
        )

    if aerosol_id == "0":
        particle_layer = {}

    elif aerosol_id in {"c", "d"}:
        particle_layer = {
            "bottom": 0.0 * ureg.m,
            "top": 2000.0 * ureg.m,
            "distribution": {"type": "uniform"},
        }

        if aerosol_id == "c":
            particle_layer["dataset"] = "govaerts_2021-continental"

        else:  # aerosol_id == "d":
            particle_layer["dataset"] = "govaerts_2021-desert"

    else:
        raise ValueError(f"Unhandled particle layer component id '{aerosol_id}'")

    if aerosol_ot == "0":
        pass
    elif aerosol_ot == "2":
        particle_layer["tau_ref"] = 0.2
    elif aerosol_ot == "6":
        particle_layer["tau_ref"] = 0.6
    else:
        raise ValueError(f"Unhandled particle layer optical thickness '{aerosol_ot}'")

    atmosphere = {
        "type": "heterogeneous",
        "molecular_atmosphere": molecular_atmosphere,
        "particle_layers": [particle_layer] if particle_layer else [],
    }

    # Final configuration
    config = {
        "canopy": canopy,
        "surface": surface,
        "atmosphere": atmosphere,
        "illumination": {
            "type": "directional",
            "zenith": 30.0 * ureg.deg,
            "azimuth": 0.0 * ureg.deg,
        },
        "measures": [
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 2) * ureg.deg,
                "azimuth": 0.0 * ureg.deg,
                "srf": "sentinel_2a-msi-4",
                "spp": spp,
            }
        ],
        "ckd_quad_config": {"type": "gauss_legendre", "ng_max": 16, "policy": "fixed"},
        "integrator": {"type": "piecewise_volpath", "moment": True},
    }

    return CanopyAtmosphereExperiment(**config)


#: Mapping of test case to a generation function for all RAMI4ATM test cases
#: available in the Eradiate regression test suite.
registry = {
    x: functools.partial(create_rami4atm, id=x)
    for x in [
        "hom00_whi_s00s_m04_z30a000_brfpp",
        "hom00_bla_a00s_m04_z30a000_brfpp",
        "hom00_rpv_e00s_m04_z30a000_brfpp",
        "hom00_rpv_0c2s_m04_z30a000_brfpp",
        "hom00_rpv_0c6s_m04_z30a000_brfpp",
        "hom00_rpv_0d2s_m04_z30a000_brfpp",
        "hom00_rpv_0d6s_m04_z30a000_brfpp",
        "hom00_rpv_sc2s_m04_z30a000_brfpp",
        "hom00_rpv_sc6s_m04_z30a000_brfpp",
        "hom00_rpv_sd2s_m04_z30a000_brfpp",
        "hom00_rpv_sd6s_m04_z30a000_brfpp",
        "hom00_rpv_ac2s_m04_z30a000_brfpp",
        "hom00_rpv_ac6s_m04_z30a000_brfpp",
        "hom00_rpv_ad2s_m04_z30a000_brfpp",
        "hom00_rpv_ad6s_m04_z30a000_brfpp",
        "hom00_lam_ec2s_m04_z30a000_brfpp",
        "hom00_rpv_ec2s_m04_z30a000_brfpp",
        "hom00_rli_ec2s_m04_z30a000_brfpp",
        "hom00_rpv_ec6s_m04_z30a000_brfpp",
        "hom00_rpv_ed2s_m04_z30a000_brfpp",
        "hom00_rpv_ed6s_m04_z30a000_brfpp",
        "hom45_lam_ec2s_m04_z30a000_brfpp",
    ]
}
