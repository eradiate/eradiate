"""
Post-processing pipeline definitions.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from hamilton.function_modifiers import (
    ResolveAt,
    config,
    extract_fields,
    parameterize,
    resolve,
    source,
    tag,
    tag_outputs,
)

from .. import logic
from ..._mode import modes
from ...scenes.illumination import (
    Illumination,
)
from ...spectral import SpectralResponseFunction
from ...spectral.grid import SpectralGrid

_MODE_IDS_CKD = set(modes(lambda x: x.is_ckd))


def _parameterize_aggregate_ckd_quad(var_name):
    # Tailored decorator for the aggregate_ckd_quad() task
    cfg = {var_name: {"raw_data": source(f"{var_name}_raw")}}
    return parameterize(**cfg)


@tag(**{"final": "true", "kind": "data"})
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_parameterize_aggregate_ckd_quad,
)
def aggregate_ckd_quad(
    mode_id: str, raw_data: xr.DataArray, spectral_grid: SpectralGrid, ckd_quads: list
) -> xr.DataArray:
    return logic.aggregate_ckd_quad(mode_id, raw_data, spectral_grid, ckd_quads, False)


def _parameterize_aggregate_ckd_quad_var(var_name):
    # Tailored decorator for the aggregate_ckd_quad_var() task
    cfg = {
        f"{var_name}_var": {"raw_data": source(f"{var_name}_var_raw")},
    }
    return parameterize(**cfg)


@tag(**{"final": "true", "kind": "data"})
@config.when(calculate_variance=True)
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_parameterize_aggregate_ckd_quad_var,
)
def aggregate_ckd_quad_var(
    mode_id: str, raw_data: xr.DataArray, spectral_grid: SpectralGrid, ckd_quads: list
) -> xr.DataArray:
    return logic.aggregate_ckd_quad(mode_id, raw_data, spectral_grid, ckd_quads, True)


def _parameterize_variance(var_name):
    # Tailored decorator for the moment2_to_variance() task
    cfg = {
        f"{var_name}_var_raw": {
            "raw_data": source(f"{var_name}_raw"),
            "raw_m2_data": source(f"{var_name}_m2_raw"),
        }
    }
    return parameterize(**cfg)


@tag(**{"kind": "data"})
@config.when(calculate_variance=True)
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_parameterize_variance,
)
def moment2_to_variance(
    raw_data: xr.DataArray, raw_m2_data: xr.DataArray, spp: xr.DataArray
) -> xr.DataArray:
    return logic.moment2_to_variance(raw_data, raw_m2_data, spp)


def _parameterize_apply_spectral_response(var_name, measure_distant):
    # Tailored decorator for the _apply_spectral_response() task

    # Add basic measure variable
    cfg = {f"{var_name}_srf": {"spectral_data": source(var_name)}}

    # Possibly add other derived spectral variables
    if var_name == "sector_radiosity":
        cfg["radiosity_srf"] = {"spectral_data": source("radiosity")}

    if measure_distant:
        cfg["irradiance_srf"] = {"spectral_data": source("irradiance")}

    return parameterize(**cfg)


@tag(**{"final": "true", "kind": "data"})
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE, decorate_with=_parameterize_apply_spectral_response
)
@config.when_in(mode_id=_MODE_IDS_CKD)
@config.when(apply_spectral_response=True)
def apply_spectral_response(
    spectral_data: xr.DataArray, srf: SpectralResponseFunction
) -> xr.DataArray:
    return logic.apply_spectral_response(spectral_data, srf)


@tag(**{"final": "true", "kind": "data"})
@config.when(var_name="sector_radiosity", measure_distant=True)
def albedo(radiosity: xr.DataArray, irradiance: xr.DataArray) -> xr.DataArray:
    return logic.compute_albedo(radiosity, irradiance)


@tag(**{"final": "true", "kind": "data"})
@config.when(
    var_name="sector_radiosity", measure_distant=True, apply_spectral_response=True
)
def albedo_srf(
    radiosity_srf: xr.DataArray, irradiance_srf: xr.DataArray
) -> xr.DataArray:
    return logic.compute_albedo(radiosity_srf, irradiance_srf)


@tag_outputs(
    **{
        "brdf": {"final": "true", "kind": "data"},
        "brf": {"final": "true", "kind": "data"},
    }
)
@config.when(var_name="radiance", measure_distant=True)
@extract_fields({"brdf": xr.DataArray, "brf": xr.DataArray})
def bidirectional_reflectance(
    radiance: xr.DataArray, irradiance: xr.DataArray, calculate_stokes: bool
) -> dict:
    result = logic.compute_bidirectional_reflectance(
        radiance, irradiance, calculate_stokes
    )
    return {k: result[k] for k in result.keys()}


@tag_outputs(
    **{
        "brdf_srf": {"final": "true", "kind": "data"},
        "brf_srf": {"final": "true", "kind": "data"},
    }
)
@config.when_in(mode_id=_MODE_IDS_CKD)
@config.when(var_name="radiance", measure_distant=True, apply_spectral_response=True)
@extract_fields({"brdf_srf": xr.DataArray, "brf_srf": xr.DataArray})
def bidirectional_reflectance_srf(
    radiance_srf: xr.DataArray, irradiance_srf: xr.DataArray, calculate_stokes: bool
) -> dict:
    result = logic.compute_bidirectional_reflectance(
        radiance_srf, irradiance_srf, calculate_stokes
    )
    return {"brdf_srf": result["brdf"], "brf_srf": result["brf"]}


@tag_outputs(
    **{
        "irradiance": {"final": "true", "kind": "data"},
        "solar_angles": {"final": "true", "kind": "coord"},
    }
)
@extract_fields({"irradiance": xr.DataArray, "solar_angles": xr.Dataset})
def extract_irradiance(
    mode_id: str, illumination: Illumination, spectral_grid: SpectralGrid
) -> dict:
    return logic.extract_irradiance(mode_id, illumination, spectral_grid)


def _tag_outputs_gather_bitmaps(var_name, calculate_variance, calculate_stokes):
    fields = {
        "spp": {"kind": "data"},
        f"{var_name}_raw": {"kind": "data"},
    }
    if calculate_variance:
        fields[f"{var_name}_m2_raw"] = {"kind": "data"}

    return tag_outputs(**fields)


def _extract_fields_gather_bitmaps(var_name, calculate_variance, calculate_stokes):
    fields = {
        # Requested sample count for the corresponding Mitsuba render pass
        "spp": xr.DataArray,
        # Raw pixel weights (equal to collected sampled count with a box filter)
        "weights_raw": xr.DataArray,
        # Main film pixel variable being processed
        f"{var_name}_raw": xr.DataArray,
        # Wavelength list, source from bitmap data
        # "wavelengths": xr.DataArray,
    }

    if calculate_variance:
        # 2nd moment of the main film
        fields[f"{var_name}_m2_raw"] = xr.DataArray

    return extract_fields(fields)


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_tag_outputs_gather_bitmaps,
)
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_extract_fields_gather_bitmaps,
)
def gather_bitmaps(
    mode_id: str,
    var_name: str,
    var_metadata: dict,
    calculate_variance: bool,
    calculate_stokes: bool,
    bitmaps: dict,
    viewing_angles: xr.Dataset,
    solar_angles: xr.Dataset,
) -> dict:
    return logic.gather_bitmaps(
        mode_id,
        var_name,
        var_metadata,
        calculate_variance,
        calculate_stokes,
        bitmaps,
        viewing_angles,
        solar_angles,
    )


@tag(**{"final": "true", "kind": "data"})
@config.when(var_name="sector_radiosity")
def radiosity(sector_radiosity: xr.DataArray) -> xr.DataArray:
    return logic.radiosity(sector_radiosity)


@config.when(apply_spectral_response=True)
def spectral_response(srf: SpectralResponseFunction) -> xr.DataArray:
    return logic.spectral_response(srf)


@tag(**{"final": "true", "kind": "coord"})
@config.when(add_viewing_angles=True)
def viewing_angles(angles: np.ndarray) -> xr.Dataset:
    return logic.viewing_angles(angles)


@tag(**{"final": "true", "kind": "data"})
@config.when(calculate_stokes=True)
def dlp(radiance: xr.DataArray) -> xr.DataArray:
    return logic.degree_of_linear_polarization(radiance)


@tag(**{"final": "true", "kind": "data"})
@config.when(calculate_stokes=True, apply_spectral_response=True)
def dlp_srf(radiance_srf: xr.DataArray) -> xr.DataArray:
    return logic.degree_of_linear_polarization(radiance_srf)
