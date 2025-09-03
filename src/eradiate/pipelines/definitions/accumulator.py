"""
Post-processing pipeline definitions.
"""

from __future__ import annotations

from typing import Callable, Union

import mitsuba as mi
import xarray as xr
from hamilton.function_modifiers import (
    ResolveAt,
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
from ...spectral.grid import SpectralGrid

_MODE_IDS_CKD = set(modes(lambda x: x.is_ckd))


def _tag_outputs_gather_bitmaps(var_name):
    fields = {
        "spp": {"kind": "data"},
        f"{var_name}_raw": {"kind": "data"},
    }
    return tag_outputs(**fields)


def _extract_fields_gather_bitmaps(var_name):
    fields = {
        # Requested sample count for the corresponding Mitsuba render pass
        "spp": xr.DataArray,
        # Main film pixel variable being processed
        f"{var_name}_raw": xr.DataArray,
        # Wavelength list, source from bitmap data
        # "wavelengths": xr.DataArray,
    }
    return extract_fields(fields)


@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_tag_outputs_gather_bitmaps,
)
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=_extract_fields_gather_bitmaps,
)
def gather_tensors(
    mode_id: str,
    var_name: str,
    var_metadata: dict,
    tensors: dict,
    tensor_to_dataarray: Union[dict, Callable[[mi.TensorXf], xr.DataArray], None],
    # viewing_angles: xr.Dataset,
    solar_angles: xr.Dataset,
) -> dict:
    return logic.gather_tensors(
        mode_id,
        var_name,
        var_metadata,
        tensors,
        tensor_to_dataarray,
        solar_angles,
    )


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
