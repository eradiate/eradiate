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

from . import logic
from .._mode import modes
from ..scenes.illumination import (
    Illumination,
)
from ..scenes.spectra import (
    Spectrum,
)
from ..spectral.spectral_set import SpectralSet

_MODE_IDS_CKD = set(modes(lambda x: x.is_ckd).keys())


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
    mode_id: str, raw_data: xr.DataArray, spectral_set: SpectralSet
) -> xr.DataArray:
    return logic.aggregate_ckd_quad(mode_id, raw_data, spectral_set)


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
def apply_spectral_response(spectral_data: xr.DataArray, srf: Spectrum) -> xr.DataArray:
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
def bidirectional_reflectance(radiance: xr.DataArray, irradiance: xr.DataArray) -> dict:
    result = logic.compute_bidirectional_reflectance(radiance, irradiance)
    return {k: result[k] for k in result.keys()}


@tag_outputs(
    **{
        "brdf_srf": {"final": "true", "kind": "data"},
        "brf_srf": {"final": "true", "kind": "data"},
    }
)
@config.when(var_name="radiance", measure_distant=True, apply_spectral_response=True)
@extract_fields({"brdf_srf": xr.DataArray, "brf_srf": xr.DataArray})
def bidirectional_reflectance_srf(
    radiance_srf: xr.DataArray, irradiance_srf: xr.DataArray
) -> dict:
    result = logic.compute_bidirectional_reflectance(radiance_srf, irradiance_srf)
    return {"brdf_srf": result["brdf"], "brf_srf": result["brf"]}


@tag_outputs(
    **{
        "irradiance": {"final": "true", "kind": "data"},
        "solar_angles": {"final": "true", "kind": "coord"},
    }
)
@extract_fields({"irradiance": xr.DataArray, "solar_angles": xr.Dataset})
def extract_irradiance(
    mode_id: str, illumination: Illumination, spectral_set: SpectralSet
) -> dict:
    return logic.extract_irradiance(mode_id, illumination, spectral_set)


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda var_name: tag_outputs(
        **{
            "spp": {"kind": "data"},
            f"{var_name}_raw": {"kind": "data"},
        }
    ),
)
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda var_name: extract_fields(
        {
            # Requested sample count for the corresponding Mitsuba render pass
            "spp": xr.DataArray,
            # Raw pixel weights (equal to collected sampled count with a box filter)
            "weights_raw": xr.DataArray,
            # Main film pixel variable being processed
            f"{var_name}_raw": xr.DataArray,
            # Wavelength list, source from bitmap data
            # "wavelengths": xr.DataArray,
        }
    ),
)
def gather_bitmaps(
    mode_id: str,
    var_name: str,
    var_metadata: dict,
    bitmaps: dict,
    viewing_angles: xr.Dataset,
    solar_angles: xr.Dataset,
) -> dict:
    return logic.gather_bitmaps(
        mode_id, var_name, var_metadata, bitmaps, viewing_angles, solar_angles
    )


@tag(**{"final": "true", "kind": "data"})
@config.when(var_name="sector_radiosity")
def radiosity(sector_radiosity: xr.DataArray) -> xr.DataArray:
    return logic.radiosity(sector_radiosity)


@config.when(apply_spectral_response=True)
def spectral_response(srf: Spectrum) -> xr.DataArray:
    return logic.spectral_response(srf)


@tag(**{"final": "true", "kind": "coord"})
@config.when(add_viewing_angles=True)
def viewing_angles(angles: np.ndarray) -> xr.Dataset:
    return logic.viewing_angles(angles)
