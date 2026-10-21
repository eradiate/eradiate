from __future__ import annotations

import datetime
import math
import warnings
from typing import Literal

import numpy as np
import pint
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from ._file_resolver import fresolver
from ..typing import PathLike
from ..units import magnitude_as, symbol, to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.deprecation import deprecated
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
            "libradtran_to_prt(): The input dataset specifies no units "
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


def _var(dims, x, standard_name: str, long_name: str, **extra) -> tuple:
    """Build an xarray variable tuple; quantities are stripped and their units
    stored in the ``units`` attribute."""
    attrs = {"standard_name": standard_name, "long_name": long_name}
    if isinstance(x, pint.Quantity):
        x, attrs["units"] = x.m, symbol(x.u)
    return dims, x, {**attrs, **extra}


@deprecated(
    deprecated_in="1.3.0",
    removed_in="2.0.0",
    details="Use ``make_prt_core_v1()`` instead.",
)
def make_aer_core_v2(*args, **kwargs):
    """Compatibility alias to :func:`.make_prt_core_v1`."""
    return make_prt_core_v1(*args, **kwargs)


# Metadata of the state dimensions inserted by Prt variants between ``w``
# and the angular / moment dimensions: (standard_name, long_name)
_PRT_STATE_DIMS = {
    "reff": ("effective_radius", "effective radius"),
    "veff": ("effective_variance", "effective variance"),
    "hum": ("relative_humidity", "relative humidity"),
}


def make_prt_v1(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: pint.Quantity,
    state_dims: dict[str, pint.Quantity] | None = None,
    nangles: np.ndarray | None = None,
    pmom: np.ndarray | None = None,
    nmom: int | np.ndarray | None = None,
    grid_res: int = 3,
    attrs: dict | None = None,
    normalize: bool = False,
    check: Literal["none", "fast", "full"] | None = None,
) -> xr.Dataset:
    """
    Create a new dataset in any variant of the Prt v1 format.

    All Prt variants share the core dimensions ``w``, ``phamat``, ``iangle``
    and ``imom``. Variants add state dimensions, inserted after ``w`` (see
    :ref:`sec-data_formats-prt_extensions`): *e.g.* Prt-PSD adds ``reff`` and
    ``veff``, Prt-Hum adds ``hum``. These are passed through ``state_dims``;
    in the shapes below, ``*state`` stands for their sizes, in the order of
    ``state_dims``. Without state dimensions, the output is in the Prt-Core
    format.

    Parameters
    ----------
    w : quantity
        Wavelength, shape (nw,).

    phamat : list of str
        Phase matrix component list, shape (nphamat,).

    mu : quantity
        Scattering angle cosine, shape (nw, \\*state, nangle).

    theta : quantity
        Scattering angle, shape (nw, \\*state, nangle).

    ext : quantity
        Extinction coefficient, shape (nw, \\*state).

    ssa : quantity
        Single-scattering albedo, shape (nw, \\*state).

    phase : quantity
        Phase matrix values, shape (nphamat, nw, \\*state, nangle).
        Integral normalized to 2 (*i.e.* ∫ p(μ) dμ = 2).

    state_dims : dict, optional
        Mapping of state dimension names to their coordinate values, each of
        shape (n,) and strictly increasing (irregular grids are supported).
        Known dimensions (``reff``, ``veff``, ``hum``) get CF metadata;
        unknown ones use their name as standard and long name.

    nangles : ndarray, optional
        Number of valid angular samples, shape (nw, \\*state), dtype int.
        When provided, entries beyond ``nangles`` in ``mu``, ``theta``, and
        ``phase`` must be NaN-padded. The ``nangles`` dataset variable is
        always written; if not provided, it is inferred by counting non-NaN
        entries in the first phase matrix component.

    pmom : ndarray, optional
        Legendre polynomials for the (1,1) phase matrix element, shape
        (nw, \\*state, nimom). If not provided, they are computed
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
        * ``ndarray`` shape ``(nw, *state)`` — count stored as-is.
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
        If a value check fails, if ``mu`` does not have one dimension per
        core and state dimension, or if a state dimension coordinate is not
        strictly increasing.

    TypeError
        If ``nmom`` is an ndarray but ``pmom`` is not provided.
    """
    state_dims = state_dims or {}

    if mu.ndim != 2 + len(state_dims):
        raise ValueError(
            f"make_prt_v1(): 'mu' must have shape (nw, "
            f"{''.join(f'n{d}, ' for d in state_dims)}nangle) -- got {mu.shape}"
        )

    for name, grid in state_dims.items():
        if np.any(np.diff(grid.m) <= 0):
            raise ValueError(f"make_prt_v1(): '{name}' must be strictly increasing")

    # nangles is always stored. Infer from the first phase matrix component when
    # not provided: count non-NaN entries along the angular dimension.
    if nangles is None:
        nangles = _count_valid(phase.m[0])

    # Valid consecutive angular sample pairs, shape (nw, *state, nangle - 1)
    valid_pairs = np.arange(mu.shape[-1] - 1) < (nangles[..., None] - 1)
    dmu = np.diff(mu.m, axis=-1)

    if check is not None and check != "none":
        bad = np.any(valid_pairs & ~(dmu > 0), axis=-1)
        if check == "fast":
            rng = np.random.default_rng(seed=0)
            sampled = np.zeros(bad.size, dtype=bool)
            sampled[
                rng.choice(bad.size, size=max(1, bad.size // 10), replace=False)
            ] = True
            bad &= sampled.reshape(bad.shape)
        if np.any(bad):
            raise ValueError(
                f"make_prt_v1(): mu is not strictly ascending at "
                f"index {tuple(np.argwhere(bad)[0].tolist())}. Sort the angular "
                "grid in ascending mu order before calling this function."
            )

    if normalize:
        # Trapezoidal integral of p11 over the valid portion of each row
        p11 = phase.m[0]
        segments = 0.5 * (p11[..., 1:] + p11[..., :-1]) * dmu
        integral = np.where(valid_pairs, segments, 0.0).sum(axis=-1)
        if np.any(integral == 0.0):
            raise ValueError(
                f"make_prt_v1(): p11 integrates to 0 at index "
                f"{tuple(np.argwhere(integral == 0.0)[0].tolist())}; cannot normalize."
            )
        # NaN padding stays NaN
        phase = phase * (2.0 / integral)[None, ..., None]

    if pmom is None:
        nmom = 129 if nmom is None else nmom
        if not isinstance(nmom, int):
            raise TypeError(
                "make_prt_v1(): 'nmom' must be an int or None when 'pmom' is "
                "not provided. Pass an ndarray only together with an explicit 'pmom'."
            )

        # Compute pmom for p_11 only. mu is ascending (-1 → +1), so theta is
        # descending (180° → 0°); compute_pmom() expects theta ascending, so
        # each slice is reversed.
        theta_deg = theta.m_as(ureg.deg)
        pmom = np.zeros((*nangles.shape, nmom))
        for idx in np.ndindex(nangles.shape):
            n = int(nangles[idx])
            pmom[idx] = _compute_pmom(
                theta_deg[idx][:n][::-1],
                phase.m[(0, *idx)][:n][::-1],
                nmom=nmom,
                grid_res=grid_res,
                coefficients=True,
                check_normalization=True,
            )
    elif nmom is None:
        # Explicit pmom without nmom: count non-NaN entries
        nmom = _count_valid(pmom)

    if isinstance(nmom, int):
        nmom = np.full(nangles.shape, nmom, dtype=np.int32)

    grid = ["w", *state_dims]
    return xr.Dataset(
        data_vars={
            "ext": _var(grid, ext, "extinction_coefficient", "extinction coefficient"),
            "ssa": _var(
                grid, ssa, "single_scattering_albedo", "single-scattering albedo"
            ),
            "phase": _var(
                ["phamat", *grid, "iangle"],
                phase,
                "phase_matrix",
                "phase matrix",
                comment="integral normalized to 2",
            ),
            "nangles": _var(
                grid, nangles, "n_angular_samples", "number of angular samples"
            ),
            "nmom": _var(
                grid, nmom, "n_legendre_polys", "number of Legendre polynomials"
            ),
            "pmom": _var(
                [*grid, "imom"],
                pmom,
                "legendre_polys",
                "Legendre polynomials",
                comment="p11 component only; including factor 2l+1",
            ),
        },
        coords={
            "w": _var("w", w, "wavelength", "wavelength"),
            "phamat": _var("phamat", phamat, "component", "phase matrix component"),
            "mu": _var(
                [*grid, "iangle"],
                mu,
                "cos_scattering_angle",
                "scattering angle cosine",
            ),
            "theta": _var(
                [*grid, "iangle"], theta, "scattering_angle", "scattering angle"
            ),
            **{
                name: _var(name, values, *_PRT_STATE_DIMS.get(name, (name, name)))
                for name, values in state_dims.items()
            },
        },
        attrs=attrs or {},
    )


def make_prt_core_v1(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: pint.Quantity,
    **kwargs,
) -> xr.Dataset:
    """
    Create a new dataset in the Prt-Core v1 format.

    This is :func:`.make_prt_v1` without state dimensions: ``mu`` and
    ``theta`` have shape (nw, nangle), ``ext`` and ``ssa`` (nw,), ``phase``
    (nphamat, nw, nangle).

    Parameters
    ----------
    w, phamat, mu, theta, ext, ssa, phase
        See :func:`.make_prt_v1`.

    **kwargs
        Forwarded to :func:`.make_prt_v1`.

    Returns
    -------
    Dataset
    """
    return make_prt_v1(w, phamat, mu, theta, ext, ssa, phase, **kwargs)


def make_prt_psd_v1(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: pint.Quantity,
    reff: pint.Quantity,
    veff: pint.Quantity,
    **kwargs,
) -> xr.Dataset:
    """
    Create a new dataset in the Prt-PSD v1 format.

    This is :func:`.make_prt_v1` with state dimensions ``reff`` and ``veff``:
    ``mu`` and ``theta`` have shape (nw, nreff, nveff, nangle), ``ext`` and
    ``ssa`` (nw, nreff, nveff), ``phase`` (nphamat, nw, nreff, nveff, nangle).

    Parameters
    ----------
    w, phamat, mu, theta, ext, ssa, phase
        See :func:`.make_prt_v1`.

    reff : quantity
        Effective radius grid, shape (nreff,), strictly increasing.

    veff : quantity
        Effective variance grid, shape (nveff,), strictly increasing.

    **kwargs
        Forwarded to :func:`.make_prt_v1`.

    Returns
    -------
    Dataset
    """
    return make_prt_v1(
        w,
        phamat,
        mu,
        theta,
        ext,
        ssa,
        phase,
        state_dims={"reff": reff, "veff": veff},
        **kwargs,
    )


def make_prt_hum_v1(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: pint.Quantity,
    hum: pint.Quantity,
    **kwargs,
) -> xr.Dataset:
    """
    Create a new dataset in the Prt-Hum v1 format.

    This is :func:`.make_prt_v1` with state dimension ``hum``: ``mu`` and
    ``theta`` have shape (nw, nhum, nangle), ``ext`` and ``ssa`` (nw, nhum),
    ``phase`` (nphamat, nw, nhum, nangle).

    Parameters
    ----------
    w, phamat, mu, theta, ext, ssa, phase
        See :func:`.make_prt_v1`.

    hum : quantity
        Relative humidity grid, shape (nhum,), strictly increasing.

    **kwargs
        Forwarded to :func:`.make_prt_v1`.

    Returns
    -------
    Dataset
    """
    return make_prt_v1(
        w, phamat, mu, theta, ext, ssa, phase, state_dims={"hum": hum}, **kwargs
    )


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
    Concatenate several Prt-PSD v1 datasets along the ``veff`` dimension.

    Parameters
    ----------
    datasets : list of xr.Dataset
        Prt-PSD v1 datasets (as produced by :func:`make_prt_psd_v1`),
        each with a ``reff``/``veff``-dimensioned layout. Every dataset may
        cover one or several ``veff`` values. All datasets must share the
        same ``w``, ``reff``, and ``phamat`` coordinates.

    Returns
    -------
    xr.Dataset
        Prt-PSD v1 dataset with a ``veff`` dimension spanning the concatenation
        of all input datasets' ``veff`` values. ``iangle``/``imom`` are
        padded to the largest size found across inputs. Only attributes common
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

    return make_prt_psd_v1(
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


def aer_v1_to_prt_core_v1(
    ds: xr.Dataset,
    phase_scale: float = 1.0,
    dtype: DTypeLike | None = None,
    update_history: bool = True,
    normalize: bool = True,
    check: Literal["none", "fast", "full"] = "full",
) -> xr.Dataset:
    """
    Convert a dataset in the :ref:`Aer v1 <sec-data_formats-aer_v1>`
    format to :ref:`Prt-Core v1 <sec-data_formats-prt_core_v1>`.

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
        over μ ∈ [-1, 1].  Passed to :func:`make_prt_core_v1`; see that
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
    theta = ureg.Quantity(np.rad2deg(np.arccos(mu)), "deg")
    mu = ureg.Quantity(mu, "dimensionless")

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
    attrs["format"] = "Prt-Core v1"
    if update_history:
        utcnow = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        history = []
        if "history" in attrs:
            history.extend(attrs["history"].split("\n"))
        history.append(f"{utcnow} - Conversion from Aer v1 to Prt-Core v1")
        attrs["history"] = "\n".join(history)

    if len(_phase_datasets) == 1 or len(_phase_datasets) == 4:
        attrs["particle_shape"] = "spherical"
    elif len(_phase_datasets) == 6:
        attrs["particle_shape"] = "spheroidal"
    else:
        raise ValueError("could not determine particle shape")

    return make_prt_core_v1(
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


def libradtran_to_prt(
    ds: PathLike | xr.Dataset,
    attrs: dict | None = None,
    fallback_units: dict[str, str] | None = None,
    normalize: bool = False,
    check: Literal["none", "fast", "full"] = "full",
    *,
    reff: pint.Quantity | float | ArrayLike | None = None,
    veff: float | None = None,
    hum: pint.Quantity | float | ArrayLike | None = None,
    squeeze: bool = False,
    tolerance: dict[str, pint.Quantity | float] | None = None,
    wbounds: tuple = (None, None),
) -> xr.Dataset:
    """
    Convert a libRadtran scattering particle property dataset to Prt v1.

    The output variant depends on the dimensions remaining after point
    selection (see ``reff`` and ``hum``): Prt-PSD v1 if a ``reff`` dimension
    remains, Prt-Hum v1 if a ``hum`` dimension remains, Prt-Core v1 otherwise.

    Parameters
    ----------
    ds : Dataset or path-like
        A dataset in the libRadtran format. If a path is passed, it is
        resolved by the file resolver and loaded into memory.

    attrs : dict, optional
        Additional metadata applied to the converted dataset.

    fallback_units : dict, optional
        A mapping that specifies units to apply to variables that are missing
        them.

    normalize : bool, default: False
        If ``True``, rescale the phase matrix so that ``p₁₁`` integrates to 2
        over μ ∈ [-1, 1].  Passed to :func:`make_prt_v1`; see that function
        for details.

    check : {"none", "fast", "full"}, default: "full"
        Apply validation checks to the output.

    reff : float or quantity or array-like, optional
        Effective radius points to select (nearest neighbour), for datasets
        with an effective radius dimension. Selection follows
        :meth:`xarray.Dataset.sel`: a scalar drops the dimension, a 1D array
        keeps it. If unset, no selection is performed and the dimension is
        kept (see ``squeeze``). Unitless values are interpreted in µm.

    veff : float, optional
        The effective variance attached to Prt-PSD v1 output, as a size-1
        ``veff`` dimension. If unset, it defaults to ``1 / (param_alpha + 3)``,
        the standard gamma-distribution relation, computed from
        ``ds.attrs["param_alpha"]``; if that attribute is missing, the output
        has no ``veff`` dimension. Ignored if the output has no ``reff``
        dimension.

    hum : float or quantity or array-like, optional
        Relative humidity points to select (nearest neighbour), for datasets
        with a humidity dimension. Same selection rules as ``reff``. Unitless
        values are interpreted in percent.

    squeeze : bool, default: False
        If ``True``, drop size-1 ``reff`` and ``hum`` dimensions on which no
        selection is performed.

    tolerance : dict, optional
        Maximum distance for the nearest-neighbour lookup, indexed by
        ``"reff"`` and ``"hum"``. Units follow the same rules as for the
        ``reff`` and ``hum`` parameters.

    wbounds : tuple, default: (None, None)
        Bounds restricting the spectral domain where conversion is performed,
        as a (min, max) pair; use ``None`` to leave a bound open. Examples:
        ``(300, 3000)``, ``(None, 3000)``, ``(0.3, 3) * ureg.micron``.
        Unitless values are interpreted in ``ucc.get("wavelength")`` units.

    Returns
    -------
    xr.Dataset

    Raises
    ------
    ValueError
        If a variable has no units and ``fallback_units`` does not provide
        them.

    Notes
    -----
    * All conversion is done in memory: very large datasets might result in
      massive converted data. In such cases, restrict the spectral domain
      using ``wbounds``.

    * The Prt-Core v1 format has a single angular discretization common to all
      phase matrix coefficients, while libRadtran's data aerosol format allows
      for a different one for each coefficient. The angular grid of the output
      dataset is constructed as the union of the libRadtran grids for all
      coefficients; all coefficients are then interpolated linearly in μ space
      on the angular points that were missing for them. The same construction
      applies independently at each ``reff`` or ``hum`` point.
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

    if not isinstance(ds, xr.Dataset):
        ds = fresolver.load_dataset(ds)

    # Select humidity and effective radius points; dimensions are renamed
    # after their coordinate variable
    tolerance = tolerance or {}
    for var, dim, value, default_units in [
        ("hum", "nhum", hum, ureg.percent),
        ("reff", "nreff", reff, ureg.micron),
    ]:
        if dim not in ds.dims:
            if value is not None:
                warnings.warn(
                    f"libradtran_to_prt(): '{var}' was passed but the dataset "
                    f"has no '{dim}' dimension; ignoring it",
                    stacklevel=2,
                )
            continue

        ds = ds.swap_dims({dim: var})

        if value is None:
            if squeeze and ds.sizes[var] == 1:
                ds = ds.isel({var: 0}).drop_vars(var)
            continue

        units = _get_units(ds, var, fallback_units)
        sel_tolerance = tolerance.get(var)
        if sel_tolerance is not None:
            sel_tolerance = magnitude_as(sel_tolerance, units, default_units)
        ds = ds.sel(
            {var: magnitude_as(value, units, default_units)},
            method="nearest",
            tolerance=sel_tolerance,
        )
        if var not in ds.dims:  # Scalar selection
            ds = ds.drop_vars(var)

    # Remaining state dimensions; libRadtran datasets have either nhum or
    # nreff, so there is at most one
    state = [d for d in ("hum", "reff") if d in ds.dims]
    state_shape = tuple(ds.sizes[d] for d in state)
    npts = math.prod(state_shape)

    # Gather wavelengths, restricted to the requested spectral domain
    units_w = _get_units(ds, "wavelen", fallback_units)
    wmin, wmax = wbounds
    if wmin is not None or wmax is not None:
        wavelen = ds["wavelen"].values
        mask = np.ones(wavelen.shape, dtype=bool)
        if wmin is not None:
            mask &= wavelen >= magnitude_as(wmin, units_w, ucc.get("wavelength"))
        if wmax is not None:
            mask &= wavelen <= magnitude_as(wmax, units_w, ucc.get("wavelength"))
        ds = ds.isel(nlam=mask)

    w = ds["wavelen"].values * units_w
    nw = len(w)

    def gather(var: str, *trailing: str) -> np.ndarray:
        # Shape (nw, npts, *trailing)
        x = ds[var].transpose("nlam", *state, *trailing).values
        return x.reshape(nw, npts, *x.shape[1 + len(state) :])

    state_dims = {d: ds[d].values * _get_units(ds, d, fallback_units) for d in state}
    if "reff" in state:
        if veff is None and "param_alpha" in ds.attrs:
            veff = 1.0 / (float(ds.attrs["param_alpha"]) + 3.0)
        if veff is not None:
            state_dims["veff"] = np.atleast_1d(veff) * ureg.dimensionless
    # Output shape of state dimensions; veff is size 1 and lands after reff
    out_shape = state_shape + ((1,) if "veff" in state_dims else ())

    # Gather extinction coefficients and albedo
    units_ext = _get_units(ds, "ext", fallback_units)
    ext_np = gather("ext")  # (nw, npts)
    units_ssa = _get_units(ds, "ssa", fallback_units)
    ssa_np = gather("ssa")  # (nw, npts)

    # Process phase function entries
    nphamat = ds.sizes["nphamat"]
    phamat = [x for x, _ in PHAMAT_TO_IDX[:nphamat]]

    # --- Build union angular grid (across all components), at every
    # (w, state) point, and interpolate onto it ---
    ntheta_np = gather("ntheta", "nphamat")  # (nw, npts, nphamat)
    theta_np = gather("theta", "nphamat", "nthetamax")  # (nw, npts, nphamat, nthetamax)
    phase_np = gather("phase", "nphamat", "nthetamax")  # (nw, npts, nphamat, nthetamax)

    union_mus: list[np.ndarray] = []
    union_phases: list[np.ndarray] = []  # each entry: (nphamat, n_union)

    for iw in range(nw):
        for ipt in range(npts):
            # Collect per-component (mu, phase), then build the union grid in one pass.
            comp_data: list[tuple[np.ndarray, np.ndarray]] = []
            for i_pm in range(nphamat):
                n = int(ntheta_np[iw, ipt, i_pm])
                comp_mu = np.cos(np.deg2rad(theta_np[iw, ipt, i_pm, :n]))
                comp_data.append((comp_mu, phase_np[iw, ipt, i_pm, :n]))

            # np.unique returns sorted, unique values (ascending); clip for
            # floating-point safety.
            mu_union = np.unique(np.concatenate([mu for mu, _ in comp_data]))
            mu_union = np.clip(mu_union, -1.0, 1.0)

            # Interpolate each component onto the union mu grid
            phase_pt = np.empty((nphamat, len(mu_union)))  # (nphamat, n_union)
            for i_pm, (comp_mu, phase_) in enumerate(comp_data):
                idx = np.argsort(comp_mu)
                phase_pt[i_pm] = np.interp(mu_union, comp_mu[idx], phase_[idx])

            union_mus.append(mu_union)
            union_phases.append(phase_pt)

    # Build NaN-padded arrays, shaped (nw, *out_shape, ...)
    nangles_out = np.array([len(m) for m in union_mus], dtype=np.int32).reshape(
        nw, *out_shape
    )
    mu_arr = _pad_ragged(union_mus).reshape(nw, *out_shape, -1)
    phase_arr = np.stack(
        [
            _pad_ragged([pt[i_pm] for pt in union_phases]).reshape(nw, *out_shape, -1)
            for i_pm in range(nphamat)
        ],
        axis=0,
    )  # (nphamat, nw, *out_shape, nthetamax_out)
    theta_arr = np.rad2deg(np.arccos(mu_arr))  # NaN propagates

    # Copy Legendre coefficients (p_11 only)
    pmom = (
        ds["pmom"]
        .isel(nphamat=0)
        .transpose("nlam", *state, "nmommax")
        .values.reshape(nw, *out_shape, -1)
    )
    nmom = _count_valid(pmom)  # (nw, *out_shape)

    return make_prt_v1(
        w=w,
        phamat=phamat,
        mu=ureg.Quantity(mu_arr, "dimensionless"),
        theta=ureg.Quantity(theta_arr, "deg"),
        ext=ext_np.reshape(nw, *out_shape) * units_ext,
        ssa=ssa_np.reshape(nw, *out_shape) * units_ssa,
        phase=ureg.Quantity(phase_arr, "1/sr"),
        state_dims=state_dims or None,
        nmom=nmom,
        pmom=pmom,
        nangles=nangles_out,
        attrs=attrs or ds.attrs,
        normalize=normalize,
        check=check,
    )


@deprecated(
    deprecated_in="1.3.0",
    removed_in="2.0.0",
    details="Use ``libradtran_to_prt()`` instead",
)
def libradtran_to_aer_core_v2(
    ds: xr.Dataset,
    attrs: dict | None = None,
    fallback_units: dict[str, str] | None = None,
    normalize: bool = False,
    check: Literal["none", "fast", "full"] = "full",
):
    """Compatibility alias to :func:`libradtran_to_prt`."""
    return libradtran_to_prt(ds, attrs, fallback_units, normalize, check)
