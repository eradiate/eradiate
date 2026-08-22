from __future__ import annotations

import datetime
from typing import Literal

import numpy as np
import pint
import xarray as xr
from numpy.typing import DTypeLike

from ..units import symbol, to_quantity
from ..units import unit_registry as ureg
from ..util.pmom import compute_pmom as _compute_pmom


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
            "libradtran_to_aer_core_v2(): The input dataset specifies no units "
            f"for variable '{var}'; this can be addressed by passing them through "
            "the 'fallback_units' parameter."
        )


def _pad_ragged(arrays: list[np.ndarray], fillval: float = np.nan) -> np.ndarray:
    # https://stackoverflow.com/a/40571482/3645374
    lens = np.array([len(a) for a in arrays])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(arrays)
    return out


def _count_valid(a: np.ndarray) -> np.ndarray:
    """Count non-NaN entries along the last axis, consistent with the
    NaN-padding convention used throughout this module."""
    return np.count_nonzero(~np.isnan(a), axis=-1).astype(np.int32)


def _maybe_cast(q: pint.Quantity, dtype: DTypeLike | None) -> pint.Quantity:
    """Cast a quantity to ``dtype``, preserving the origin dtype when
    ``dtype`` is ``None``."""
    return q.astype(dtype) if dtype else q


def make_aer_core_v2(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: pint.Quantity,
    reff: pint.Quantity | None = None,
    veff: pint.Quantity | None = None,
    nangles: np.ndarray | None = None,
    pmom: np.ndarray | None = None,
    nmom: int | np.ndarray | None = None,
    grid_res: int = 3,
    attrs: dict | None = None,
    normalize: bool = False,
    check: Literal["none", "fast", "full"] | None = None,
) -> xr.Dataset:
    """
    Create a new dataset in the Aer-Core v2 format, or the Prt v1 format
    when ``reff``/``veff`` are provided.

    Parameters
    ----------
    w : quantity
        Wavelength, shape (nw,).

    phamat : list of str
        Phase matrix component list, shape (nphamat,).

    mu : quantity
        Scattering angle cosine, shape (nw, [nreff, nveff, ]nangle).

    theta : quantity
        Scattering angle, shape (nw, [nreff, nveff, ]nangle).

    ext : quantity
        Extinction coefficient, shape (nw, [nreff, nveff]).

    ssa : quantity
        Single-scattering albedo, shape (nw, [nreff, nveff]).

    phase : quantity
        Phase matrix values, shape (nphamat, nw, [nreff, nveff, ]nangle).
        Integral normalized to 2 (*i.e.* ∫ p(μ) dμ = 2).

    reff, veff : quantity, optional
        Effective radius and variance grids, shape (nreff,) and (nveff,).
        When set, the dataset is built in the Prt v1 format: all
        other arrays gain the corresponding ``reff``/``veff`` dimensions,
        between ``w`` and the angular/moment dimension. Both must be set
        together, or neither.

    nangles : ndarray, optional
        Number of valid angular samples, shape (nw, [nreff, nveff]), dtype
        int. When provided, entries beyond ``nangles`` in ``mu``, ``theta``,
        and ``phase`` must be NaN-padded. The ``nangles`` dataset variable is
        always written; if not provided, it is inferred by counting non-NaN
        entries in the first phase matrix component.

    pmom : ndarray, optional
        Legendre polynomials for the (1,1) phase matrix element, shape
        (nw, [nreff, nveff, ]nimom). If not provided, they are computed
        automatically from ``phase`` and ``theta`` using
        :func:`.compute_pmom` with ``coefficients=True``.

    nmom : int or ndarray, optional
        Controls the number of Legendre polynomials. The ``nmom`` dataset
        variable is always written when ``pmom`` is present.

        * ``None`` (default) — when ``pmom`` is not provided, compute 129
          coefficients and store their count as a uniform ``nmom`` array;
          when ``pmom`` is provided explicitly, infer counts by counting
          non-NaN entries in the first phase matrix component, consistent
          with the NaN-padding convention.
        * ``int`` — scalar moment count; used as the target for automatic
          computation and stored as a uniform ``nmom`` array.
        * ``ndarray`` shape ``(nw, [nreff, nveff])`` — count stored as-is.
          ``pmom`` must be provided explicitly when ``nmom`` is an ndarray.

    grid_res : int, default: 3
        Angular grid resolution passed to :func:`.compute_pmom` when ``pmom``
        is computed automatically. ``3`` uses the input grid with the exact
        BDE method (recommended). See :func:`.compute_pmom` for all options.

    attrs : dict
        Dataset attributes.

    normalize : bool, default: False
        If ``True``, rescale the phase matrix so that ``p₁₁`` integrates to 2
        over μ ∈ [-1, 1].  All phase matrix components are multiplied by the
        same factor (2 / ∫p₁₁ dμ), preserving their relative magnitudes.
        When ``False`` (default), the caller is responsible for providing a
        pre-normalized ``phase`` array.

    check : {"none", "fast", "full"}, optional
        Controls validation of the ``mu`` angular grid sort order.

        * ``None`` or ``"none"`` — no check (default).
        * ``"fast"`` — sampling-based check: validates a random 10 % of
          entries (at least one). Intended as a cheap guard during bulk
          production runs.
        * ``"full"`` — validates every entry.

        A ``ValueError`` is raised if any sampled row is not strictly
        monotonically increasing in μ. Only the valid (non-NaN) portion of
        each row is checked (bounded by ``nangles``).

    Returns
    -------
    Dataset

    Raises
    ------
    ValueError
        If a value check fails, if ``nmom`` is an ndarray but ``pmom`` is
        not provided, or if only one of ``reff``/``veff`` is provided.
    """
    if (reff is None) != (veff is None):
        raise ValueError(
            "make_aer_core_v2(): 'reff' and 'veff' must be provided together, "
            "or not at all"
        )
    extra_dims = [] if reff is None else ["reff", "veff"]

    # nangles is always stored. Infer from the first phase matrix component when
    # not provided: count non-NaN entries along the angular dimension.
    if nangles is None:
        nangles = _count_valid(phase.m[0])

    if check is not None and check != "none":
        mu_vals = mu.m  # (nw, [nreff, nveff, ]nangle)
        indices = list(np.ndindex(mu_vals.shape[:-1]))

        if check == "fast":
            rng = np.random.default_rng(seed=0)
            n_sample = max(1, len(indices) // 10)
            indices = [
                indices[i]
                for i in rng.choice(len(indices), size=n_sample, replace=False)
            ]

        for idx in indices:
            n = int(nangles[idx])
            row = mu_vals[idx][:n]
            if not np.all(np.diff(row) > 0):
                raise ValueError(
                    f"make_aer_core_v2(): mu is not strictly ascending at "
                    f"index {idx}. Sort the angular grid in ascending "
                    "mu order before calling this function."
                )

    if normalize:
        phase_vals = phase.m.copy()  # (nphamat, nw, [nreff, nveff, ]nangle)
        mu_vals = mu.m  # (nw, [nreff, nveff, ]nangle)
        for idx in np.ndindex(mu_vals.shape[:-1]):
            n = int(nangles[idx])
            mu_pt = mu_vals[idx][:n]
            phase_pt = phase_vals[(slice(None), *idx)]  # (nphamat, nangle)
            integral = np.trapezoid(phase_pt[0, :n], mu_pt)
            if integral == 0.0:
                raise ValueError(
                    f"make_aer_core_v2(): p11 integrates to 0 at index "
                    f"{idx}; cannot normalize."
                )
            phase_pt[:, :n] *= 2.0 / integral
        phase = phase_vals * phase.u

    if pmom is None:
        # Determine the scalar moment count for automatic computation.
        if nmom is None:
            _nmom_scalar = 129
        elif isinstance(nmom, int):
            _nmom_scalar = nmom
        else:
            raise ValueError(
                "make_aer_core_v2(): 'nmom' must be an int or None when 'pmom' is "
                "not provided. Pass an ndarray only together with an explicit 'pmom'."
            )

        # Compute pmom for p_11 (index 0) only.
        # mu in the dataset is ascending (-1 → +1), so theta is descending
        # (180° → 0°). compute_pmom() expects theta ascending (0° → 180°),
        # so we reverse each slice.
        theta_deg = theta.to("deg").magnitude  # (nw, [nreff, nveff, ]nangle)
        phase_vals = phase.m  # (nphamat, nw, [nreff, nveff, ]nangle)
        leading_shape = theta_deg.shape[:-1]

        pmom = np.zeros(leading_shape + (_nmom_scalar,))
        for idx in np.ndindex(leading_shape):
            n = int(nangles[idx])
            theta_pt = theta_deg[idx][:n][::-1]  # ascending 0° → 180°
            phase_pt = phase_vals[(0, *idx)][:n][::-1]  # p_11 component
            pmom[idx] = _compute_pmom(
                theta_pt,
                phase_pt,
                nmom=_nmom_scalar,
                grid_res=grid_res,
                coefficients=True,
                check_normalization=True,
            )
        nmom = np.full(leading_shape, _nmom_scalar, dtype=np.int32)

    # nmom is always stored alongside pmom.
    # Normalize any remaining scalar or inferred form to a per-entry array.
    if nmom is None:
        # pmom was provided explicitly without nmom: count non-NaN entries,
        # consistent with the NaN-padding convention.
        nmom = _count_valid(pmom)
    elif isinstance(nmom, int):
        nmom = np.full(np.shape(nangles), nmom, dtype=np.int32)

    data_vars = {
        "ext": (
            ["w", *extra_dims],
            ext.m,
            {
                "standard_name": "extinction_coefficient",
                "long_name": "extinction coefficient",
                "units": symbol(ext.u),
            },
        ),
        "ssa": (
            ["w", *extra_dims],
            ssa.m,
            {
                "standard_name": "single_scattering_albedo",
                "long_name": "single-scattering albedo",
                "units": symbol(ssa.u),
            },
        ),
        "phase": (
            ["phamat", "w", *extra_dims, "iangle"],
            phase.m,
            {
                "standard_name": "phase_matrix",
                "long_name": "phase matrix",
                "units": symbol(phase.u),
                "comment": "integral normalized to 2",
            },
        ),
    }

    data_vars["nangles"] = (
        ["w", *extra_dims],
        nangles,
        {
            "standard_name": "n_angular_samples",
            "long_name": "number of angular samples",
        },
    )

    data_vars["nmom"] = (
        ["w", *extra_dims],
        nmom,
        {
            "standard_name": "n_legendre_polys",
            "long_name": "number of Legendre polynomials",
        },
    )

    data_vars["pmom"] = (
        ["w", *extra_dims, "imom"],
        pmom,
        {
            "standard_name": "legendre_polys",
            "long_name": "Legendre polynomials",
            "comment": "p11 component only; including factor 2l+1",
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
            ["w", *extra_dims, "iangle"],
            mu.m,
            {
                "standard_name": "cos_scattering_angle",
                "long_name": "scattering angle cosine",
                "units": symbol(mu.u),
            },
        ),
        "theta": (
            ["w", *extra_dims, "iangle"],
            theta.m,
            {
                "standard_name": "scattering_angle",
                "long_name": "scattering angle",
                "units": symbol(theta.u),
            },
        ),
    }

    if reff is not None:
        coords["reff"] = (
            "reff",
            reff.m,
            {
                "standard_name": "effective_radius",
                "long_name": "effective radius",
                "units": symbol(reff.u),
            },
        )
        coords["veff"] = (
            "veff",
            veff.m,
            {
                "standard_name": "effective_variance",
                "long_name": "effective variance",
                "units": symbol(veff.u),
            },
        )

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs or {})


def _shared_attrs(datasets: list[xr.Dataset]) -> dict:
    """Attrs common to all *datasets*, by key and value; the rest are dropped."""
    ref_attrs = datasets[0].attrs
    return {
        key: value
        for key, value in ref_attrs.items()
        if all(ds.attrs.get(key, ref_attrs) == value for ds in datasets[1:])
    }


def concat_particles_veff(datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Concatenate several Prt v1 datasets along the ``veff`` dimension.

    Parameters
    ----------
    datasets : list of xr.Dataset
        Prt v1 datasets (as produced by :func:`make_aer_core_v2` with
        ``reff``/``veff`` set), each with a ``reff``/``veff``-dimensioned
        layout. Every dataset may cover one or several ``veff`` values. All
        datasets must share the same ``w``, ``reff``, and ``phamat``
        coordinates.

    Returns
    -------
    xr.Dataset
        Prt v1 dataset with a ``veff`` dimension spanning the concatenation
        of all input datasets' ``veff`` values. ``iangle``/``imom`` are
        padded to the largest size found across inputs. Only attrs common
        to all input datasets are retained.

    Raises
    ------
    ValueError
        If the datasets do not share the same ``w``, ``reff``, or ``phamat``
        coordinates.
    """
    ref = datasets[0]

    for ds in datasets[1:]:
        if not np.array_equal(ds["w"].values, ref["w"].values):
            raise ValueError("All datasets must share the same 'w' coordinate.")
        if not np.array_equal(ds["reff"].values, ref["reff"].values):
            raise ValueError("All datasets must share the same 'reff' coordinate.")
        if not np.array_equal(ds["phamat"].values, ref["phamat"].values):
            raise ValueError("All datasets must share the same 'phamat' coordinate.")

    n_iangle = max(ds.sizes["iangle"] for ds in datasets)
    n_imom = max(ds.sizes["imom"] for ds in datasets)

    def pad_last(a: np.ndarray, n: int) -> np.ndarray:
        pad_width = [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])]
        return np.pad(a, pad_width, mode="constant", constant_values=np.nan)

    mu = np.concatenate(
        [pad_last(ds["mu"].values, n_iangle) for ds in datasets], axis=2
    )
    theta = np.concatenate(
        [pad_last(ds["theta"].values, n_iangle) for ds in datasets], axis=2
    )
    phase = np.concatenate(
        [pad_last(ds["phase"].values, n_iangle) for ds in datasets], axis=3
    )
    pmom = np.concatenate(
        [pad_last(ds["pmom"].values, n_imom) for ds in datasets], axis=2
    )
    ext = np.concatenate([ds["ext"].values for ds in datasets], axis=2)
    ssa = np.concatenate([ds["ssa"].values for ds in datasets], axis=2)
    nangles = np.concatenate([ds["nangles"].values for ds in datasets], axis=2)
    nmom = np.concatenate([ds["nmom"].values for ds in datasets], axis=2)
    veff = np.concatenate([ds["veff"].values for ds in datasets])

    return make_aer_core_v2(
        w=to_quantity(ref["w"]),
        phamat=list(ref["phamat"].values),
        mu=mu * ureg(ref["mu"].attrs["units"]),
        theta=theta * ureg(ref["theta"].attrs["units"]),
        ext=ext * ureg(ref["ext"].attrs["units"]),
        ssa=ssa * ureg(ref["ssa"].attrs["units"]),
        phase=phase * ureg(ref["phase"].attrs["units"]),
        reff=to_quantity(ref["reff"]),
        veff=veff * ureg(ref["veff"].attrs["units"]),
        nangles=nangles,
        pmom=pmom,
        nmom=nmom,
        attrs=_shared_attrs(datasets),
    )


def aer_v1_to_aer_core_v2(
    ds: xr.Dataset,
    phase_scale: float = 1.0,
    dtype: DTypeLike | None = None,
    update_history: bool = True,
    normalize: bool = True,
    check: Literal["none", "fast", "full"] = "full",
) -> xr.Dataset:
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

    dtype : dtype-like, optional
        Dtype of floating-point variables. If unset, the origin dtype is
        preserved.

    update_history : bool, default: True
        If ``True``, update the ``history`` attribute of the dataset with a new
        entry mentioning the conversion timestamp.

    normalize : bool, default: True
        If ``True``, rescale the phase matrix so that ``p₁₁`` integrates to 2
        over μ ∈ [-1, 1].  Passed to :func:`make_aer_core_v2`; see that
        function for details.

    check : {"none", "fast", "full"}, default: "full"
        Apply validation checks to the output.

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

    w = _maybe_cast(to_quantity(ds["w"]), dtype)
    ext = _maybe_cast(to_quantity(ds["sigma_t"]), dtype)
    ssa = _maybe_cast(to_quantity(ds["albedo"]), dtype)

    nangles = ds.sizes["mu"]
    nw = ds.sizes["w"]

    # Output is always in ascending mu order (-1 → +1).
    mu_1d = ds["mu"].values
    sort_idx = np.argsort(mu_1d)

    mu = np.broadcast_to(mu_1d[sort_idx], (nw, nangles)).copy()
    if dtype:
        mu = mu.astype(dtype)
    mu = mu * ureg("dimensionless")
    theta = (np.arccos(mu.m) * ureg("rad")).to("deg")

    _phase_datasets = {}
    for ij, (i, j) in PHAMAT_TO_IDX:
        if i in ds["phase"]["i"].values and j in ds["phase"]["j"].values:
            _phase_datasets[ij] = ds["phase"].sel(i=i, j=j, drop=True)

    # Remove duplicates (indicate particle symmetry)
    for x, y in [("11", "22"), ("33", "44")]:
        if (
            x in _phase_datasets
            and y in _phase_datasets
            and np.allclose(_phase_datasets[x], _phase_datasets[y])
        ):
            del _phase_datasets[y]

    phase = (
        _maybe_cast(
            to_quantity(
                xr.concat(_phase_datasets.values(), dim="phamat").transpose(
                    "phamat", "w", "mu"
                )
            ),
            dtype,
        )
        * phase_scale
    )

    # Reorder angle dimension to ascending mu
    phase = phase[:, :, sort_idx]

    attrs = ds.attrs.copy()
    attrs["format"] = "Aer-Core v2"
    if update_history:
        utcnow = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        history = []
        if "history" in attrs:
            history.extend(attrs["history"].split("\n"))
        history.append(f"{utcnow} - Conversion from Aer v1 to Aer-Core v2")
        attrs["history"] = "\n".join(history)

    if len(_phase_datasets) == 1 or len(_phase_datasets) == 4:
        attrs["particle_shape"] = "spherical"
    elif len(_phase_datasets) == 6:
        attrs["particle_shape"] = "spheroidal"
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
        normalize=normalize,
        check=check,
    )


def libradtran_to_aer_core_v2(
    ds: xr.Dataset,
    attrs: dict | None = None,
    fallback_units: dict[str, str] | None = None,
    normalize: bool = False,
    check: Literal["none", "fast", "full"] = "full",
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

    normalize : bool, default: False
        If ``True``, rescale the phase matrix so that ``p₁₁`` integrates to 2
        over μ ∈ [-1, 1].  Passed to :func:`make_aer_core_v2`; see that
        function for details.

    check : {"none", "fast", "full"}, default: "full"
        Apply validation checks to the output.

    Returns
    -------
    xr.Dataset

    Notes
    -----
    * The Aer-Core v2 format has a single angular discretization common to all
      phase matrix coefficients, while libRadtran's data aerosol format allows
      for a different one for each coefficient. The angular grid of the output
      dataset is constructed as the union of the libRadtran grids for all
      coefficients; all coefficients are then interpolated linearly in μ space
      on the angular points that were missing for them.
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
    nphamat = ds.sizes["nphamat"]
    phamat = [x for x, _ in PHAMAT_TO_IDX[:nphamat]]

    # --- Build union angular grid (across all components) and interpolate onto it ---
    ntheta_np = ds["ntheta"].values  # (nw, nphamat)
    theta_np = ds["theta"].values  # (nw, nphamat, nthetamax)
    phase_np = ds["phase"].values  # (nw, nphamat, nthetamax)

    union_mus: list[np.ndarray] = []
    union_phases: list[np.ndarray] = []  # each entry: (nphamat, n_union)

    for iw in range(nw):
        # Collect per-component (mu, phase), then build the union grid in one pass.
        comp_data: list[tuple[np.ndarray, np.ndarray]] = []
        for i_pm in range(nphamat):
            n = int(ntheta_np[iw, i_pm])
            comp_mu = np.cos(np.deg2rad(theta_np[iw, i_pm, :n]))
            comp_data.append((comp_mu, phase_np[iw, i_pm, :n]))

        # np.unique returns sorted, unique values (ascending); clip for
        # floating-point safety.
        mu_union = np.unique(np.concatenate([mu for mu, _ in comp_data]))
        mu_union = np.clip(mu_union, -1.0, 1.0)

        # Interpolate each component onto the union mu grid
        phase_iw = np.empty((nphamat, len(mu_union)))
        for i_pm, (comp_mu, phase_) in enumerate(comp_data):
            idx = np.argsort(comp_mu)
            phase_iw[i_pm] = np.interp(mu_union, comp_mu[idx], phase_[idx])

        union_mus.append(mu_union)
        union_phases.append(phase_iw)

    # Build NaN-padded 2-D arrays
    nangles_out = np.array([len(m) for m in union_mus], dtype=np.int32)

    mu_arr = _pad_ragged(union_mus)  # (nw, nthetamax_out)
    phase_arr = np.full((nphamat, nw, mu_arr.shape[1]), np.nan)
    for iw in range(nw):
        n_ = nangles_out[iw]
        phase_arr[:, iw, :n_] = union_phases[iw]

    mu = mu_arr * ureg("dimensionless")
    phase = phase_arr * ureg("1 / sr")
    theta = np.rad2deg(np.arccos(mu_arr)) * ureg("deg")  # NaN propagates

    # Copy Legendre coefficients (p_11 only)
    pmom = ds["pmom"].isel(nphamat=0).transpose("nlam", "nmommax").values
    nmom = _count_valid(pmom)

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
        nangles=nangles_out,
        attrs=attrs,
        normalize=normalize,
        check=check,
    )
