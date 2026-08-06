"""
RAMI4ATM benchmark scenarios.

Scenarios are defined here, in the package, rather than in the test suite, so
that they can also be driven by the performance benchmarks (see
``benchmarks/benchmarks/bench_rami4atm.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import attrs
import numpy as np
import xarray as xr

from ..regression import RegressionTest, RMSETest, ZTest
from ...attrs import define, documented
from ...experiments import CanopyAtmosphereExperiment, Experiment
from ...units import unit_registry as ureg

#: Mapping of RAMI4ATM band identifiers to Eradiate SRF identifiers.
SRF_IDS = {
    "m02": "sentinel_2a-msi-2",
    "m03": "sentinel_2a-msi-3",
    "m04": "sentinel_2a-msi-4",
    "m8a": "sentinel_2a-msi-8a",
    "m11": "sentinel_2a-msi-11",
    "m12": "sentinel_2a-msi-12",
}


def srf_id(case_id: str) -> str:
    """
    Identifier of the spectral response function used by a RAMI4ATM case.

    Parameters
    ----------
    case_id : str
        Identifier of the test case (see :func:`create_toa` for its grammar).

    Returns
    -------
    str
        SRF identifier, resolvable by the file resolver as ``srf/<id>.nc``.

    Raises
    ------
    ValueError
        If the case ID uses an unhandled band identifier.
    """
    band_id = case_id.split("_")[3]

    try:
        return SRF_IDS[band_id]
    except KeyError:
        raise ValueError(f"Unhandled band id '{band_id}'") from None


def create_toa(case_id: str, spp: int, padding: int = 5) -> CanopyAtmosphereExperiment:
    """
    Create an :class:`.Experiment` simulating a top-of-atmosphere RAMI4ATM
    benchmarking case.

    Parameters
    ----------
    case_id : str
        Identifier of the test case (see notes for supported values).

    spp : int
        Default measure sample count.

    padding : int, default: 5
        Instance padding radius (total number of instances is equal to
        ``(padding × 2 + 1)²``). Relevant only to cases featuring a canopy.

    Returns
    -------
    .CanopyAtmosphereExperiment

    Raises
    ------
    ValueError
        If the passed case ID contains unhandled parameter values.

    Notes
    -----
    Case IDs are structured as follows (see also the
    `RAMI4ATM case naming convention reference\
    <https://rami-benchmark.jrc.ec.europa.eu/_www/format_RAMI4ATM.php?strPhase=RAMI4ATM>`__::

    <canopy_id>_<surface_id>_<atmosphere_id>_<band_id>_<illumination_id>_<measure_id>

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
          - ``m02``, ``m03``, ``m04``, ``m8a``, ``m11``, ``m12``
            (Sentinel-2 / MSI bands)
        * - ``illumination_id``
          - ``z30a000`` (30° zenith, 0° azimuth)
        * - ``measure_id``
          - ``brfpp`` (TOA BRF in the principal plane)
    """
    canopy_id, surface_id, atmosphere_id, _, _, _ = case_id.split("_")

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
    if surface_id == "bla":
        surface = {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.0},
        }
    elif surface_id == "whi":
        surface = {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 1.0},
        }
    elif surface_id == "lam":
        surface = {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.02806},
        }
    elif surface_id == "rpv":
        surface = {
            "type": "rpv",
            "rho_0": {"type": "uniform", "value": 0.017051},
            "k": {"type": "uniform", "value": 0.95},
            "g": {"type": "uniform", "value": -0.1},
            "rho_c": {"type": "uniform", "value": 0.017051},
        }
    elif surface_id == "rli":
        surface = {
            "type": "rtls",
            "f_iso": {"type": "uniform", "value": 0.032171},
            "f_vol": {"type": "uniform", "value": -0.002886},
            "f_geo": {"type": "uniform", "value": 0.001949},
        }
    else:
        raise ValueError(f"Unhandled surface id '{surface_id}'")

    # Atmosphere setup
    molecular_id, aerosol_id, aerosol_ot_id, profile_id = atmosphere_id

    if molecular_id == "0":
        molecular_atmosphere = None

    elif molecular_id in {"a", "s", "e"}:
        molecular_atmosphere = {
            "type": "molecular",
            "thermoprops": {
                "identifier": "afgl_1986-us_standard",
                "z": np.arange(0, 120.05, 0.05) * ureg.km,
            },
            "absorption_data": "monotropa",
            "has_absorption": molecular_id in {"a", "e"},
            "has_scattering": molecular_id in {"s", "e"},
        }

    else:
        raise ValueError(
            f"Unhandled molecular atmosphere component id '{molecular_id}'"
        )

    if aerosol_id == "0":
        particle_layer = {}

    elif aerosol_id in {"c", "d"}:
        particle_layer = {
            "bottom": 0.0 * ureg.m,
            "top": 2000.0 * ureg.m,
            "distribution": {"type": "uniform"},
            "particle_properties": (
                "govaerts_2021-continental"
                if aerosol_id == "c"
                else "govaerts_2021-desert"
            ),
        }

    else:
        raise ValueError(f"Unhandled particle layer component id '{aerosol_id}'")

    if aerosol_ot_id == "0":
        pass
    elif aerosol_ot_id in {"2", "6"}:
        if not particle_layer:
            raise ValueError(
                f"Optical thickness id '{aerosol_ot_id}' requires a particle "
                f"layer, but the aerosol id is '{aerosol_id}' (none)"
            )
        particle_layer["tau_ref"] = 0.2 if aerosol_ot_id == "2" else 0.6
    else:
        raise ValueError(
            f"Unhandled particle layer optical thickness '{aerosol_ot_id}'"
        )

    if profile_id != "s":
        raise ValueError(f"Unhandled atmospheric profile id '{profile_id}'")

    atmosphere = {
        "type": "heterogeneous",
        "molecular_atmosphere": molecular_atmosphere,
        "particle_layers": [particle_layer] if particle_layer else [],
    }

    return CanopyAtmosphereExperiment(
        canopy=canopy,
        surface=surface,
        atmosphere=atmosphere,
        illumination={
            "type": "directional",
            "zenith": 30.0 * ureg.deg,
            "azimuth": 0.0 * ureg.deg,
        },
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 2) * ureg.deg,
                "azimuth": 0.0 * ureg.deg,
                "srf": srf_id(case_id),
                "spp": spp,
                "target": target,
            }
        ],
        ckd_quad_config={"type": "gauss_legendre", "ng_max": 16, "policy": "fixed"},
        integrator={"type": "piecewise_volpath", "moment": True},
    )


def create_boa(case_id: str, spp: int) -> list[CanopyAtmosphereExperiment]:
    """
    Create the :class:`.Experiment` objects simulating the bottom-of-atmosphere
    counterpart of a RAMI4ATM benchmarking case.

    Bottom-of-atmosphere reflectance factors are ratios: they need both the
    quantity leaving the target and the quantity a perfectly diffuse surface
    would have reflected in its place. This function therefore returns four
    experiments, measuring, in order:

    1. the upwelling radiance of the target;
    2. the upwelling radiance of a white reference patch;
    3. the radiosity of the target;
    4. the radiosity of a white reference patch.

    Parameters
    ----------
    case_id : str
        Identifier of the top-of-atmosphere case the scene is taken from (see
        :func:`create_toa` for its grammar).

    spp : int
        Default measure sample count.

    Returns
    -------
    list of .CanopyAtmosphereExperiment
    """
    exp_base = create_toa(case_id, spp=spp)
    canopy_id = case_id.split("_")[0]

    if canopy_id == "hom00":
        white_patch = {
            "factory": "shape",
            "type": "rectangle",
            "center": [0.0, 0.0, 0.01] * ureg.m,
            "edges": [1.0, 1.0] * ureg.m,
            "bsdf": {"type": "lambertian", "reflectance": 1.0},
        }
        target = [0.0, 0.0, 0.01] * ureg.m
    else:
        white_patch = {
            "factory": "shape",
            "type": "rectangle",
            "center": [0.0, 0.0, 2.025] * ureg.m,
            "edges": [5.0, 5.0] * ureg.m,
            "bsdf": {"type": "lambertian", "reflectance": 1.0},
        }
        target = {
            "type": "rectangle",
            "xmin": -2.5 * ureg.m,
            "xmax": 2.5 * ureg.m,
            "ymin": -2.5 * ureg.m,
            "ymax": 2.5 * ureg.m,
            "z": 2.025 * ureg.m,
        }

    exp_radiance = attrs.evolve(
        exp_base,
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 1) * ureg.deg,
                "azimuth": 0.0 * ureg.deg,
                "srf": srf_id(case_id),
                "spp": spp,
                "ray_offset": 0.05,
                "target": target,
            }
        ],
    )
    exp_radiance_white = attrs.evolve(
        exp_radiance, extra_objects={"boa_white_reference_patch": white_patch}
    )

    radiosity_measure = {
        "type": "distantflux",
        "srf": srf_id(case_id),
        "spp": spp,
        "ray_offset": 0.05,
        "target": target,
    }
    exp_radiosity = attrs.evolve(exp_radiance, measures=[radiosity_measure])
    exp_radiosity_white = attrs.evolve(exp_radiance_white, measures=[radiosity_measure])

    return [exp_radiance, exp_radiance_white, exp_radiosity, exp_radiosity_white]


def _ratio_variance(
    numerator: xr.DataArray,
    numerator_var: xr.DataArray,
    denominator: xr.DataArray,
    denominator_var: xr.DataArray,
) -> xr.DataArray:
    """
    Variance of the ratio of two independent estimates, to first order.

    Parameters
    ----------
    numerator, denominator : DataArray
        Estimates the ratio is formed of.

    numerator_var, denominator_var : DataArray
        Variance of the corresponding estimate.

    Returns
    -------
    DataArray
        Variance of ``numerator / denominator``.

    Notes
    -----
    Delta method: the ratio is linearised around the estimates, which assumes
    that their relative standard errors are small. The two experiments are
    rendered independently, so their covariance is zero.
    """
    return (
        numerator_var / denominator**2 + denominator_var * numerator**2 / denominator**4
    )


def _total_radiosity_var(result: xr.Dataset) -> xr.DataArray:
    """
    Variance of the radiosity recorded by a distant flux measure.

    Radiosity is the sum of the sector radiosities over the film, so its
    variance is the sum of theirs, neglecting inter-pixel covariance.
    """
    return result.sector_radiosity_srf_var.sum(dim=("x_index", "y_index"))


def postprocess_boa(results: list[xr.Dataset]) -> xr.Dataset:
    """
    Assemble the bottom-of-atmosphere reflectance factors from the results of
    the experiments created by :func:`create_boa`.

    Parameters
    ----------
    results : list of Dataset
        Results of the four experiments, in the order :func:`create_boa`
        returns them.

    Returns
    -------
    Dataset
        Dataset holding the hemispherical-directional (``hdrf``) and
        bi-hemispherical (``bhr``) reflectance factors, their variances, and
        the four measurements they are derived from. Every variable is
        SRF-weighted; the ``_srf`` suffix is dropped for readability.
    """
    radiance, radiance_white, radiosity, radiosity_white = results

    result = xr.Dataset(
        {
            "radiance_target": radiance.radiance_srf,
            "radiance_target_var": radiance.radiance_srf_var,
            "radiance_white": radiance_white.radiance_srf,
            "radiance_white_var": radiance_white.radiance_srf_var,
            "radiosity_target": radiosity.radiosity_srf,
            "radiosity_target_var": _total_radiosity_var(radiosity),
            "radiosity_white": radiosity_white.radiosity_srf,
            "radiosity_white_var": _total_radiosity_var(radiosity_white),
        }
    )

    result["hdrf"] = result.radiance_target / result.radiance_white
    result["hdrf_var"] = _ratio_variance(
        result.radiance_target,
        result.radiance_target_var,
        result.radiance_white,
        result.radiance_white_var,
    )

    result["bhr"] = result.radiosity_target / result.radiosity_white
    result["bhr_var"] = _ratio_variance(
        result.radiosity_target,
        result.radiosity_target_var,
        result.radiosity_white,
        result.radiosity_white_var,
    )

    return result


def _single_result(results: list[xr.Dataset]) -> xr.Dataset:
    """
    Default postprocessing: a single-experiment case yields a single result.
    """
    return results[0]


@define
class Case:
    """
    Definition of a RAMI4ATM benchmarking case.

    A case bundles everything needed to run a scenario and to decide whether
    its results are acceptable, so that a consumer (*e.g.* regression test suite
    or the performance benchmarks) needs no knowledge of the scenario itself.
    """

    id: str = documented(
        attrs.field(),
        doc="Case identifier, following the RAMI4ATM naming convention (see "
        ":func:`create_toa`).",
        type="str",
    )

    make_experiments: Callable[..., list[Experiment]] = documented(
        attrs.field(),
        doc="Callable building the experiments the case is made of. It takes "
        "the sample count as its ``spp`` keyword argument.",
        type="callable",
    )

    tests: list[RegressionTest] = documented(
        attrs.field(factory=list),
        doc="Criteria the comparison against the reference must satisfy. They "
        "all share a single reference dataset.",
        type="list of :class:`.RegressionTest`",
        default="[]",
    )

    postprocess: Callable[[list[xr.Dataset]], xr.Dataset] = documented(
        attrs.field(default=_single_result),
        doc="Callable turning the results of the case's experiments into the "
        "dataset the criteria are applied to.",
        type="callable",
        default="select the single result",
    )


def _toa_experiments(case_id: str, spp: int) -> list[CanopyAtmosphereExperiment]:
    """
    Adapt the single-experiment :func:`create_toa` to the :class:`Case`
    interface.
    """
    return [create_toa(case_id, spp=spp)]


def _toa_case(case_id: str, threshold: float = 0.005) -> Case:
    """
    Build a standard top-of-atmosphere case, compared against its reference
    with a Z-test.

    The tested variable is the band-integrated radiance: that is the quantity
    the scenario is meant to hold invariant. Comparing the per-bin ``radiance``
    instead makes the verdict depend on how the CKD quadrature distributes the
    signal across bins, which is an implementation detail.
    """
    return Case(
        id=case_id,
        make_experiments=functools.partial(_toa_experiments, case_id),
        tests=[ZTest(threshold, variable="radiance_srf")],
    )


def _boa_case(toa_case_id: str, threshold: float) -> Case:
    """
    Build the bottom-of-atmosphere counterpart of a top-of-atmosphere case. Its
    reflectance factors are compared against their reference with an RMSE
    criterion.
    """
    return Case(
        id=toa_case_id.replace("brfpp", "boa"),
        make_experiments=functools.partial(create_boa, toa_case_id),
        tests=[
            RMSETest(threshold, variable="hdrf"),
            RMSETest(threshold, variable="bhr"),
        ],
        postprocess=postprocess_boa,
    )


#: RAMI4ATM benchmarking cases covered by the Eradiate regression test suite,
#: indexed by case identifier.
CASES = {
    case.id: case
    for case in [
        *[
            _toa_case(case_id)
            for case_id in [
                "hom00_whi_s00s_m04_z30a000_brfpp",
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
        ],
        _toa_case("hom00_bla_sd2s_m03_z30a000_brfpp", threshold=0.05),
        _boa_case("hom00_whi_s00s_m04_z30a000_brfpp", threshold=2e-3),
        _boa_case("hom00_rpv_e00s_m04_z30a000_brfpp", threshold=2e-5),
    ]
}
