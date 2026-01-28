import functools

import attr
import numpy as np
import xarray as xr

from ..regression import RMSETest, ZTest
from ...experiments import CanopyAtmosphereExperiment
from ...units import unit_registry as ureg


def create_rami4atm_toa(
    case: str, spp: int, padding: int = 5
) -> tuple[str, list[CanopyAtmosphereExperiment]]:
    """
    Create an :class:`.Experiment` to simulate one of the RAMI4ATM benchmarking
    cases.

    Parameters
    ----------
    case : str
        Identifier of the test case (see notes for supported values).

    spp : int
        Default measure sample count.

    Returns
    -------
    tuple
        A tuple containing the srf identifier and a list of
        :class:`.Experiment` objects.

    Raises
    ------
    ValueError
        If the passed case ID contains unhandled parameter values.

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
    canopy_id, surf_id, atm_id, band_id, illumination_id, measure_id = case.split("_")

    # Canopy setup
    if canopy_id == "hom00":
        canopy = None
        target = [0.0, 0.0, 0.0]
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
            "padding": padding,
        }
        target = {
            "type": "rectangle",
            "xmin": -2.5 * ureg.m,
            "xmax": 2.5 * ureg.m,
            "ymin": -2.5 * ureg.m,
            "ymax": 2.5 * ureg.m,
            "z": 2.0 * ureg.m,
        }
    else:
        raise ValueError(f"Unhandled canopy id '{canopy_id}'")

    # Surface setup
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

    srf_id = {
        "m02": "sentinel_2a-msi-2",
        "m03": "sentinel_2a-msi-3",
        "m04": "sentinel_2a-msi-4",
        "m8a": "sentinel_2a-msi-8a",
        "m11": "sentinel_2a-msi-11",
        "m12": "sentinel_2a-msi-12",
    }[band_id]

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
                "srf": srf_id,
                "spp": spp,
                "target": target,
            }
        ],
        "ckd_quad_config": {"type": "gauss_legendre", "ng_max": 16, "policy": "fixed"},
        "integrator": {"type": "piecewise_volpath", "moment": True},
    }

    return srf_id, [CanopyAtmosphereExperiment(**config)]


def create_rami4atm_boa(
    toa_case: str, spp: int
) -> tuple[str, list[CanopyAtmosphereExperiment]]:
    """
    Create BOA experiments from Rami4ATM case id

    Return 4 experiments for each BOA case, computing observed the upwelling
    radiance and radiosity for the target, and ones that would have been
    reflected by a perfectly diffuse surface.
    """

    ctor = registry[toa_case]["constructor"]

    srf_id, (exp0,) = ctor(spp=spp)

    if "hom00" in toa_case:
        extra_objects = {
            "boa_white_reference_patch": {
                "factory": "shape",
                "type": "rectangle",
                "center": [0, 0, 0.01],
                "edges": [1, 1],
                "bsdf": {"type": "lambertian", "reflectance": 1.0},
            }
        }
        target = [0, 0, 0.01]
    else:
        extra_objects = {
            "boa_white_reference_patch": {
                "factory": "shape",
                "type": "rectangle",
                "center": [0, 0, 2.025],
                "edges": [5, 5],
                "bsdf": {"type": "lambertian", "reflectance": 1.0},
            }
        }
        target = {
            "type": "rectangle",
            "xmin": -2.5,
            "xmax": 2.5,
            "ymin": -2.5,
            "ymax": 2.5,
            "z": 2.025,
        }

    exp1 = attr.evolve(
        exp0,
        measures=[
            {
                "type": "mdistant",
                "spp": spp,
                "ray_offset": 0.05,
                "srf": srf_id,
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 1),
                "zeniths_units": "degree",
                "azimuth": 0.0,
                "azimuth_units": "degree",
                "target": target,
            }
        ],
    )
    exp2 = attr.evolve(exp1, extra_objects=extra_objects)
    dflux = {
        "type": "distantflux",
        "ray_offset": 0.05,
        "target": target,
        "srf": srf_id,
        "spp": spp,
    }
    exp3 = attr.evolve(exp1, measures=[dflux])
    exp4 = attr.evolve(exp2, measures=[dflux])

    return srf_id, [exp1, exp2, exp3, exp4]


def _calculate_boa_reflectance_factor_var(
    da1: xr.Dataset,
    var1: xr.Dataset,
    da2: xr.Dataset,
    var2: xr.Dataset,
    srf: xr.Dataset,
) -> xr.Dataset:
    # Handle numerical variance estimates, assuming:
    #  - Gaussian nature
    #  - Null covariance of different experiments or pixels
    #  - Local linearity of radiance and radiosity
    #  - Scalar SRF

    srf_weight = srf.srf.interp(w=da1.w.values).fillna(1e-18)
    srf_sum = srf_weight.sum()

    S1 = da1.sum() / srf_sum
    S2 = da2.sum() / srf_sum
    v_S1 = (var1 * srf_weight**2).sum(dim="w") / (srf_weight**2).sum()
    v_S2 = (var2 * srf_weight**2).sum(dim="w") / (srf_weight**2).sum()

    d_R_d_S1 = 1 / S2
    d_R_d_S2 = -S1 / S2**2

    v_R = d_R_d_S1**2 * v_S1 + d_R_d_S2**2 * v_S2

    return v_R


def postprocess_boa_cases(results: list[xr.Dataset], srf: xr.Dataset) -> xr.Dataset:
    result1, result2, result3, result4 = results

    result_hdrf = result1.radiance_srf / result2.radiance_srf
    result_bhr = result3.radiosity_srf / result4.radiosity_srf

    # Combine a reference dataset
    result = result1.rename(
        {"radiance_srf": "radiance_srf1", "radiance_var": "radiance_var1"}
    )
    result["radiance_srf2"] = result2.radiance_srf
    result["radiance_var2"] = result2.radiance_var
    result["radiosity_srf3"] = result3.radiosity_srf
    result["radiosity_var3"] = result3.sector_radiosity_var
    result["radiosity_srf4"] = result4.radiosity_srf
    result["radiosity_var4"] = result4.sector_radiosity_var
    result["hdrf"] = result_hdrf
    result["bhr"] = result_bhr

    result["hdrf_var"] = _calculate_boa_reflectance_factor_var(
        result1.radiance,
        result1.radiance_var,
        result2.radiance,
        result2.radiance_var,
        srf,
    )
    result["bhr_var"] = _calculate_boa_reflectance_factor_var(
        result3.radiosity,
        result3.sector_radiosity_var,
        result4.radiosity,
        result4.sector_radiosity_var,
        srf,
    )

    return result


#: Mapping of test case to a generation function for all RAMI4ATM test cases
#: available in the Eradiate regression test suite.

registry = {
    case: {
        "constructor": functools.partial(create_rami4atm_toa, case=case),
        "threshold": 0.005,
    }
    for case in [
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


registry["hom00_bla_sd2s_m03_z30a000_brfpp"] = {
    "constructor": functools.partial(
        create_rami4atm_toa, case="hom00_bla_sd2s_m03_z30a000_brfpp"
    ),
    "test": ZTest,
    "threshold": 0.05,
    "variables": ["radiance"],
}

registry["hom00_whi_s00s_m04_z30a000_boa"] = {
    "constructor": functools.partial(
        create_rami4atm_boa, toa_case="hom00_whi_s00s_m04_z30a000_brfpp"
    ),
    "test": RMSETest,
    "threshold": 2e-3,
    "variables": ["hdrf", "bhr"],
    "postprocess": postprocess_boa_cases,
}

registry["hom00_rpv_e00s_m04_z30a000_boa"] = {
    "constructor": functools.partial(
        create_rami4atm_boa, toa_case="hom00_rpv_e00s_m04_z30a000_brfpp"
    ),
    "test": RMSETest,
    "threshold": 2e-5,
    "variables": ["hdrf", "bhr"],
    "postprocess": postprocess_boa_cases,
}
