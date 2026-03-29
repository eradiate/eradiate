from __future__ import annotations

import datetime

import numpy as np
import pint
import xarray as xr
from axsdb.math import interp1d

from ..units import symbol, to_quantity
from ..units import unit_registry as ureg


def _convert_units(value: str):
    if value == "per cent":
        return "percent"
    return value


def _get_units(ds, var, fallback_units=None):
    if "units" in ds[var].attrs:
        units = ds[var].attrs["units"]
        if units == "-":
            units = ""
        return ureg.Unit(_convert_units(units))
    elif fallback_units is not None and var in fallback_units:
        return ureg.Unit(_convert_units(fallback_units[var]))
    else:
        raise ValueError(
            "load_aerosol_libradtran(): The input dataset specifies no units "
            f"for variable '{var}'; this can be addressed by passing them through "
            "the 'fallback_units' parameter."
        )


def make_aer_core_v2(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: np.ndarray,
    pmom: np.ndarray | None = None,
    nmom: np.ndarray | None = None,
    attrs: dict | None = None,
) -> xr.Dataset:
    """
    Create a new dataset in the Aer-Core v2 format.

    Parameters
    ----------
    w : quantity
        Wavelength, shape (nw,).

    phamat : list of str
        Phase matrix component list, shape (nphamat,).

    mu : quantity
        Scattering angle cosine, shape (nw, nangle).

    theta : quantity
        Scattering angle, shape (nw, nangle).

    ext : quantity
        Extinction coefficient, shape (nw,).

    ssa : quantity
        Single-scattering albedo, shape (nw,).

    phase : quantity
        Phase matrix values, shape (nphamat, nw, nangle).
        Integral normalized to 2 (*i.e.* ∫ p(μ) dμ = 2).

    pmom : ndarray, optional
        Legendre polynomials, shape (nw, nimom).

    nmom : ndarray, optional
        Number of Legendre polynomials, shape (nw,).

    attrs : dict
        Dataset attributes.

    Returns
    -------
    Dataset
    """
    data_vars = {
        "ext": (
            "w",
            ext.m,
            {
                "standard_name": "extinction_coefficient",
                "long_name": "extinction coefficient",
                "units": symbol(ext.u),
            },
        ),
        "ssa": (
            "w",
            ssa.m,
            {
                "standard_name": "single_scattering_albedo",
                "long_name": "single-scattering albedo",
                "units": symbol(ssa.u),
            },
        ),
        "phase": (
            ["phamat", "w", "iangle"],
            phase.m,
            {
                "standard_name": "phase_matrix",
                "long_name": "phase matrix",
                "units": symbol(phase.u),
                "comment": "integral normalized to 2",
            },
        ),
    }

    if nmom is not None:
        data_vars["nmom"] = (
            "w",
            nmom,
            {
                "standard_name": "n_legendre_polys",
                "long_name": "number of Legendre polynomials",
            },
        )

    if pmom is not None:
        data_vars["pmom"] = (
            ["phamat", "w", "imom"],
            pmom,
            {
                "standard_name": "legendre_polys",
                "long_name": "Legendre polynomials",
                "comment": "including factor 2l+1",
            },
        )

    coords = {
        "w": (
            "w",
            w.m,
            {
                "standard_name": "wavelength",
                "long_name": "wavelength",
                "units": symbol(w.u),
            },
        ),
        "phamat": (
            "phamat",
            phamat,
            {
                "standard_name": "component",
                "long_name": "phase matrix component",
            },
        ),
        "mu": (
            ["w", "iangle"],
            mu.m,
            {
                "standard_name": "cos_scattering_angle",
                "long_name": "scattering angle cosine",
                "units": symbol(mu.u),
            },
        ),
        "theta": (
            ["w", "iangle"],
            theta.m,
            {
                "standard_name": "scattering_angle",
                "long_name": "scattering angle",
                "units": symbol(theta.u),
            },
        ),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs or {})


def aer_v1_to_aer_core_v2(ds: xr.Dataset, phase_scale: float = 1.0) -> xr.Dataset:
    """
    Convert a dataset in the :ref:`Aer v1 <sec-data-formats-aer_v1>`
    format to :ref:`Aer-Core v2 <sec-data-formats-aer_core_v2>`.

    Parameters
    ----------
    ds : Dataset
        A dataset in the Aer v1 format.

    phase_scale : float, default: 1.0
        A scaling factor that is applied to change the normalization of phase
        function values.

    Returns
    -------
    Dataset
    """

    # This list maps phamat values to (i, j) pairs. The order follows libRadtran's
    # layout (first, coefficients for spherical particles; then additional coefficients
    # for spheroidal particles)
    PHAMAT_TO_IDX = [
        # Coefficients for spherical particles
        ("11", (0, 0)),
        ("12", (0, 1)),
        ("33", (2, 2)),
        ("34", (2, 3)),
        # Additional for spheroidal particles
        ("22", (1, 1)),
        ("44", (3, 3)),
    ]

    w = to_quantity(ds["w"]).astype("float32")
    ext = to_quantity(ds["sigma_t"]).astype("float32")
    ssa = to_quantity(ds["albedo"]).astype("float32")
    nangles = ds.sizes["mu"]
    nw = ds.sizes["w"]
    mu = np.broadcast_to(ds["mu"].values.astype("float32"), (nw, nangles))
    mu = mu * ureg("dimensionless")
    theta = (np.acos(mu) * ureg("rad")).to("deg").astype("float32")

    _phase_datasets = {}
    for ij, (i, j) in PHAMAT_TO_IDX:
        try:
            _phase_datasets[ij] = ds["phase"].sel(i=i, j=j, drop=True)
        except KeyError:
            pass

    # Remove duplicates (indicate particle symmetry)
    for x, y in [("11", "22"), ("33", "44")]:
        try:
            if np.allclose(_phase_datasets[x], _phase_datasets[y]):
                del _phase_datasets[y]
        except KeyError:
            pass

    phase = (
        to_quantity(
            xr.concat(_phase_datasets.values(), dim="phamat").transpose(
                "phamat", "w", "mu"
            )
        ).astype("float32")
        * phase_scale
    )

    attrs = ds.attrs.copy()
    utcnow = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    attrs["history"] += f"\n{utcnow} - Conversion from Aer v1 to Aer-Core v2"

    if len(_phase_datasets) == 1 or len(_phase_datasets) == 4:
        attrs["particle_shape"] = "sphere"
    elif len(_phase_datasets) == 6:
        attrs["particle_shape"] = "spheroid"
    else:
        raise ValueError("could not determine particle shape")

    return make_aer_core_v2(
        w=w,
        phamat=list(_phase_datasets.keys()),
        mu=mu,
        theta=theta,
        ext=ext,
        ssa=ssa,
        phase=phase,
        attrs=attrs,
    )


def libradtran_to_aer_core_v2(
    ds: xr.Dataset,
    attrs: dict | None = None,
    fallback_units: dict[str, str] | None = None,
) -> xr.Dataset:
    """
    Convert a dataset in the libRadtran scattering particle property format to
    the :ref:`Aer-Core v2 <sec-data-formats-aer_core_v2>` format.

    Parameters
    ----------
    ds : Dataset
        A dataset in the libRadtran format. No additional dimensions such as the
        effective radius or humidity are considered.

    attrs : dict, optional
        Additional metadata applied to the converted dataset.

    fallback_units : dict, optional
        A mapping that specifies units to apply to variables that are missing
        them.

    Returns
    -------
    xr.Dataset

    Notes
    -----
    The adaptive angular grid is preserved and refined to feature a constant
    grid point count. This does not change the size of the data in memory, but
    can change the size of the stored data on disk depending on the chosen
    compression method.
    """
    PHAMAT_TO_IDX = [
        # Coefficients for spherical particles
        ("11", 0),
        ("12", 1),
        ("33", 2),
        ("34", 3),
        # Additional for spheroidal particles
        ("22", 4),
        ("44", 5),
    ]

    # Gather wavelengths
    units = _get_units(ds, "wavelen", fallback_units)
    w = ds["wavelen"].values * units
    nw = len(w)

    # Gather extinction coefficients
    units = _get_units(ds, "ext", fallback_units)
    ext = ds["ext"].values * units

    # Gather albedo
    units = _get_units(ds, "ssa", fallback_units)
    ssa = ds["ssa"].values * units

    # Process phase function entries
    nangles = ds.sizes["nthetamax"]
    nphamat = ds.sizes["nphamat"]
    phamat = [x for x, _ in PHAMAT_TO_IDX[:nphamat]]

    # --- Allocate result arrays ---
    phase = np.full((nphamat, nw, nangles), np.nan)
    theta = np.full((nw, nangles), np.nan)

    # --- For each wavelength, refine the scattering angle grid
    #     and interpolate the phase function on it ---
    for iw in range(nw):
        # Refine scattering angle grid, interpolating nans linearly
        ntheta_ = int(ds["ntheta"].isel(nlam=iw, nphamat=0).values)
        theta_ = (
            ds["theta"].isel(nlam=iw, nphamat=0).values[:ntheta_]
        )  # keep only mesh for (1,1) component
        theta_refined = np.full((nangles,), np.nan)
        idx_dst = np.linspace(0, nangles - 1, ntheta_, dtype="int32")
        theta_refined[idx_dst] = theta_

        for i, j in zip(idx_dst[:-1], idx_dst[1:]):
            interpolated = np.linspace(
                theta_refined[i], theta_refined[j], j - i + 1, dtype=theta_refined.dtype
            )[1:-1]
            theta_refined[i + 1 : j] = interpolated

        # Sort input angles in ascending order
        idx_sort = np.argsort(theta_)
        phase_ = ds["phase"].isel(nlam=iw).values[:, idx_sort]
        theta_ = theta_[idx_sort]

        # Resample phase function using high-performance 1D interpolator with spectator dims support
        phase_refined = interp1d(theta_, phase_, theta_refined)

        # Store resampled angular grid and phase function
        theta[iw, :] = theta_refined
        phase[:, iw, :] = phase_refined

    # TODO: Check if interpolation in mu space wouldn't be better
    mu = np.cos(np.deg2rad(theta)) * ureg("dimensionless")
    phase *= ureg("1 / sr")
    theta *= ureg("deg")

    # Copy Legendre coefficients
    pmom = ds["pmom"].transpose("nphamat", "nlam", "nmommax").values
    nmom = np.full_like(w, -1, dtype="int32")
    for iw in range(nw):
        nmom[iw] = np.count_nonzero(~np.isnan(pmom[0, iw, :]))

    # Add metadata
    attrs = attrs or ds.attrs

    return make_aer_core_v2(
        w=w,
        phamat=phamat,
        mu=mu,
        theta=theta,
        ext=ext,
        ssa=ssa,
        phase=phase,
        nmom=nmom,
        pmom=pmom,
        attrs=attrs,
    )
