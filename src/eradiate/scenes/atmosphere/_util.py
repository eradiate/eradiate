from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from ._core import AbstractHeterogeneousAtmosphere
from ._heterogeneous import HeterogeneousAtmosphere
from ...cfconventions import ATTRIBUTES
from ...spectral import (
    CKDQuadConfig,
    CKDSpectralGrid,
    MonoSpectralGrid,
    MonoSpectralIndex,
)
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg


def eval_transmittance_mono(
    atmosphere: AbstractHeterogeneousAtmosphere,
    spectral_grid: MonoSpectralGrid,
    interaction: Literal["extinction", "absorption", "scattering"] = "extinction",
) -> xr.DataArray:
    """
    Compute the per-column transmittance of an atmosphere instance in
    monochromatic mode.

    Parameters
    ----------
    atmosphere : .AbstractHeterogeneousAtmosphere
        Atmosphere for which transmittance is evaluated.

    spectral_grid : .MonoSpectralGrid
        The spectral grid on which transmittance is evaluated.

    interaction : {"extinction", "absorption", "scattering"}, optional, default: "extinction"
        The interaction type for which transmittance is evaluated.

    Returns
    -------
    transmittance : DataArray
        Shape ``(x_cells, y_cells, w)``.
    """
    wavelengths = spectral_grid.wavelengths
    n_cells_x, n_cells_y, _ = atmosphere.geometry.grid.shape
    transmittance = np.full((n_cells_x, n_cells_y, wavelengths.size), np.nan)

    for i, w in tqdm(
        list(enumerate(spectral_grid.wavelengths)),
        desc="Evaluating transmittance (mono)",
    ):
        si = MonoSpectralIndex(w=w)
        transmittance[:, :, i] = atmosphere.eval_transmittance(
            si=si, interaction=interaction
        ).m_as(ureg.dimensionless)

    w_u = ucc.get("wavelength")
    result = xr.DataArray(
        transmittance,
        dims=["x_cells", "y_cells", "w"],
        coords={
            "w": (
                "w",
                wavelengths.m_as(w_u),
                {**ATTRIBUTES["radiation_wavelength"], **{"units": symbol(w_u)}},
            )
        },
        attrs={"units": "1", "long_name": "transmittance"},
    )
    return result


def eval_transmittance_ckd(
    atmosphere: AbstractHeterogeneousAtmosphere,
    spectral_grid: CKDSpectralGrid,
    ckd_quad_config: CKDQuadConfig | None = None,
    interaction: Literal["extinction", "absorption", "scattering"] = "extinction",
) -> xr.DataArray:
    """
    Compute the per-column transmittance of an atmosphere instance in CKD mode.

    Parameters
    ----------
    atmosphere : .AbstractHeterogeneousAtmosphere
        Atmosphere for which transmittance is evaluated.

    spectral_grid : .CKDSpectralGrid
        The spectral grid on which transmittance is evaluated.

    ckd_quad_config : CKDQuadConfig, optional
        CKD quadrature rule generation configuration. If unset, a conservative
        default is used (Gauss-Legendre, fixed, with 16 g-points).

    interaction : {"extinction", "absorption", "scattering"}, optional, default: "extinction"
        The interaction type for which transmittance is evaluated.

    Returns
    -------
    transmittance : DataArray
        Shape ``(x_cells, y_cells, w)``.
    """
    ckd_quad_config_default = {
        "type": "gauss_legendre",
        "ng_max": 16,
        "policy": "fixed",
    }
    if ckd_quad_config is None:
        ckd_quad_config = ckd_quad_config_default

    if isinstance(ckd_quad_config, dict):
        ckd_quad_config = {**ckd_quad_config_default, **ckd_quad_config}
        ckd_quad_config = CKDQuadConfig(**ckd_quad_config)

    # Check if a molecular absorption database is available
    abs_db = None
    if isinstance(atmosphere, HeterogeneousAtmosphere):
        abs_db = atmosphere.absorption_data
    else:
        pass

    n_cells_x, n_cells_y, _ = atmosphere.geometry.grid.shape

    transmittance = {}
    sis = list(spectral_grid.walk_indices(ckd_quad_config, abs_db))

    for si in tqdm(sis, desc="Evaluating transmittance (CKD)"):
        transmittance[si.as_hashable] = atmosphere.eval_transmittance(
            si=si, interaction=interaction
        ).m_as(ureg.dimensionless)

    index = pd.MultiIndex.from_tuples(transmittance.keys(), names=("w", "g"))
    da = xr.DataArray(
        np.stack(list(transmittance.values())),
        dims=["si", "x_cells", "y_cells"],
        coords={"si": index},
    ).unstack("si")

    # For each bin, compute quadrature per column and store the result
    wavelengths = spectral_grid.wavelengths
    w_u = ucc.get("wavelength")
    result = np.full((n_cells_x, n_cells_y, wavelengths.size), np.nan)

    for i, (w, quad) in enumerate(spectral_grid.walk_quads(ckd_quad_config)):
        values_at_nodes = da.sel(w=w.m).values

        # Rationale: Avoid using xarray's indexing in this loop for
        # performance reasons (wrong data indexing method will result in
        # 10x+ speed reduction)
        result[:, :, i] = 0.5 * np.einsum("g,xyg->xy", quad.weights, values_at_nodes)

    result = xr.DataArray(
        result,
        dims=["x_cells", "y_cells", "w"],
        coords={
            "w": (
                "w",
                wavelengths.m_as(w_u),
                {**ATTRIBUTES["radiation_wavelength"], **{"units": symbol(w_u)}},
            )
        },
        attrs={"units": "1", "long_name": "transmittance"},
    )

    return result
