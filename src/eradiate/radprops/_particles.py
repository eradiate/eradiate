from __future__ import annotations

import heapq
from os import PathLike
from typing import Any

import attrs
import numpy as np
import pint
import pinttrs
import xarray as xr
from axsdb.math import interp1d

from .. import converters
from ..attrs import define, documented
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import summary_repr


def _validate_shape(value: Any) -> str:
    valid = {"spherical", "spheroidal"}
    if value not in valid:
        raise ValueError(
            f"Unrecognized particle shape {value!r} (valid values are {valid})."
        )
    return value


# TODO: If _rdp1d_log becomes a performance bottleneck (e.g. with grids up to
#   10k points called frequently), consider rewriting the heap loop with Numba
#   (@njit(cache=True)). The algorithm maps well to scalar JIT compilation and
#   could yield a 50–100x speedup over pure Python with the same logic.
def _rdp1d_log(mu: np.ndarray, values: np.ndarray, n_out: int) -> np.ndarray:
    """
    Decimate a 1D phase function using Ramer-Douglas-Peucker (RDP) in log space.

    Selects ``n_out`` indices from ``mu`` that best preserve the shape of
    ``values`` under piecewise-linear interpolation in log(values) vs mu space.
    Endpoints are always retained.

    The algorithm is iterative: it starts with only the two endpoints, then
    greedily inserts the point with the largest log-space deviation from the
    current piecewise-linear approximation until the target count is reached.

    Parameters
    ----------
    mu : ndarray
        Scattering angle cosines, shape ``(n,)``, sorted ascending.

    values : ndarray
        Phase function values, shape ``(n,)``. Must be strictly positive.

    n_out : int
        Number of output points. Must satisfy ``2 <= n_out <= len(mu)``.

    Returns
    -------
    ndarray
        Sorted array of ``n_out`` indices into ``mu``/``values``.

    Notes
    -----
    When ``values`` has multiple rows (multiple phase matrix components),
    the maximum log-space error across all rows is used to select insertion
    points, so accuracy is preserved for all components simultaneously.
    """
    n = len(mu)

    if n_out >= n:
        return np.arange(n)

    log_values = np.log(np.maximum(values, 1e-300))  # shape (n,) or (n_phamat, n)

    # Priority queue entries: (-max_error, left_idx, right_idx, best_mid_idx)
    # n is typically small (< 2000), so pure Python heapq is fast enough.

    # Track which indices are currently selected (endpoints always included)
    selected = [False] * n
    selected[0] = True
    selected[n - 1] = True
    count = 2

    def _best_split(i_l: int, i_r: int) -> tuple[float, int]:
        """Return (max_log_error, argmax_idx) for interior points of segment."""
        if i_r - i_l < 2:
            return 0.0, i_l
        mu_l, mu_r = mu[i_l], mu[i_r]
        idxs = np.arange(i_l + 1, i_r)
        t = (mu[idxs] - mu_l) / (mu_r - mu_l)
        if log_values.ndim == 1:
            log_interp = log_values[i_l] + t * (log_values[i_r] - log_values[i_l])
            errors = np.abs(log_values[idxs] - log_interp)
        else:
            log_interp = log_values[:, i_l : i_l + 1] + t * (
                log_values[:, i_r : i_r + 1] - log_values[:, i_l : i_l + 1]
            )
            errors = np.abs(log_values[:, idxs] - log_interp).max(axis=0)
        best_local = int(errors.argmax())
        return float(errors[best_local]), int(idxs[best_local])

    # Initialize heap with the single segment [0, n-1]
    err, best = _best_split(0, n - 1)
    heap = [(-err, 0, n - 1, best)]

    while count < n_out and heap:
        neg_err, i_l, i_r, mid = heapq.heappop(heap)
        if selected[mid]:
            # Stale entry: mid was already inserted via another segment; skip.
            continue
        selected[mid] = True
        count += 1
        # Push the two new sub-segments
        for sl, sr in [(i_l, mid), (mid, i_r)]:
            if sr - sl >= 2:
                e, b = _best_split(sl, sr)
                heapq.heappush(heap, (-e, sl, sr, b))

    return np.where(selected)[0]


def _upsample_mu(mu: np.ndarray, n_out: int) -> np.ndarray:
    """
    Upsample a mu grid to ``n_out`` points by inserting uniformly-spaced
    interior points into each gap, with the number of points per gap allocated
    proportionally to gap size.

    All original points are preserved. The non-uniform (physics-motivated)
    density of the original grid is maintained — in particular the dense
    sampling near the forward-scattering peak.

    Parameters
    ----------
    mu : ndarray
        Sorted scattering angle cosines, shape ``(n_mu,)``.

    n_out : int
        Target number of points. If n_out <= n_mu, this function is a no-op.

    Returns
    -------
    ndarray
        Sorted array of ``n_out`` mu values containing all original points.
    """
    n = len(mu)
    n_missing = n_out - n
    if n_missing <= 0:
        return mu

    gaps = np.diff(mu)

    # Distribute n_missing points proportionally to gap size.
    # Floors may leave a small remainder; give it to the gaps with the largest
    # fractional parts so the total is exactly n_missing.
    raw = gaps * (n_missing / gaps.sum())
    counts = np.floor(raw).astype(int)
    remainder = int(n_missing - counts.sum())
    if remainder > 0:
        counts[np.argpartition(raw - counts, -remainder)[-remainder:]] += 1

    # Vectorized point generation: for each active gap i with counts[i] = k,
    # insert k uniformly-spaced interior points at
    #   mu[i] + j * gaps[i] / (k + 1),  j = 1 ... k
    active = np.where(counts > 0)[0]
    k = counts[active]
    mu_l = mu[active]
    g = gaps[active]

    # Build the j-index array [1 .. k[0], 1 .. k[1], ...] without a Python loop
    j_flat = np.ones(k.sum(), dtype=int)
    if len(k) > 1:
        j_flat[np.cumsum(k[:-1])] -= k[:-1]
    j = np.cumsum(j_flat)

    new_pts = np.repeat(mu_l, k) + j * np.repeat(g, k) / np.repeat(k + 1, k)
    return np.sort(np.concatenate([mu, new_pts]))


_resolve_keyword = converters.resolve_keyword(lambda x: f"aerosol/{x}.nc")
_resolve_path = converters.resolve_path


@define
class ParticleProperties:
    """
    An interface to scattering particle radiative properties.

    This class manages scattering particle radiative properties and implements
    interpolation routines to produce kernel-level input data.
    """

    # TODO: Validate data format
    data: xr.Dataset = documented(
        attrs.field(
            validator=attrs.validators.instance_of(xr.Dataset), repr=summary_repr
        ),
        doc="Single-scattering property dataset.",
        type="xarray.Dataset",
        init_type="xarray.Dataset",
    )

    # The following fields are used to cache properties
    # -- Wavelength
    _w: pint.Quantity | None = pinttrs.field(
        default=None, units=ucc.deferred("wavelength"), init=False, repr=False
    )
    # -- Phase function
    _phase: xr.DataArray | None = attrs.field(default=None, init=False, repr=False)
    # -- Extinction coefficient
    _ext: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    # -- Single-scattering albedo
    _ssa: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    # -- Scattering coefficient
    _scat: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    # -- Absorption coefficient
    _abs: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    # -- Legendre polynomial expansion coefficients
    _pmom: xr.DataArray | None = attrs.field(default=None, init=False, repr=False)
    # -- Shape metadata
    _particle_shape: str | None = attrs.field(default=None, init=False, repr=True)
    # -- Fixed mu grid flag
    _has_fixed_mu_grid: bool | None = attrs.field(default=None, init=False, repr=False)

    def __attrs_post_init__(self):
        # Resolve all cached attributes
        for attr in [
            "w",
            "phase",
            "ext",
            "ssa",
            "scat",
            "abs",
            "pmom",
            "particle_shape",
            "has_fixed_mu_grid",
        ]:
            getattr(self, attr)

    @classmethod
    def convert(cls, value: Any) -> ParticleProperties:
        """
        Conversion logic:

        * If value is a ParticleProperties instance, return it.
        * If value is a string, attempt resolving it as a dataset keyword.
        * If value is a path-like, load the dataset into memory and close the file.
        * If value is a Dataset, initialize a ParticleProperties with it.

        Notes
        -----
        ``xr.load_dataset`` (eager) is used instead of ``xr.open_dataset`` (lazy)
        so that the underlying file handle is closed immediately after reading. With
        lazy loading, the handle stays open until the dataset is garbage-collected.
        If construction later raises (*e.g.* a validator rejects the arguments), the
        partial object is kept alive by pytest's traceback capture and the handle is
        never closed. This leads to handle accumulation and eventual C-level crashes
        in the HDF5/netCDF4 library.
        """
        if isinstance(value, ParticleProperties):
            return value

        if isinstance(value, xr.Dataset):
            return ParticleProperties(value)

        if isinstance(value, str):
            value = _resolve_keyword(value)

        if isinstance(value, (str, PathLike)):
            value = _resolve_path(value)

            try:
                value = xr.load_dataset(value)
            except ValueError as e:
                raise ValueError(
                    f"{value!r} does not point to a file that can be resolved "
                    "by the file resolver "
                ) from e

            return ParticleProperties(value)

        raise TypeError(
            f"could not convert value of type {type(value).__name__} to ParticleProperties"
        )

    @property
    def w(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Wavelength array, cached to minimize overhead.
        """
        if self._w is None:
            self._w = to_quantity(self.data["w"]).to(ucc.get("wavelength"))
        return self._w

    @property
    def abs(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Absorption coefficient array, cached to minimize overhead.
        """
        if self._abs is None:
            self._abs = self.ext * (1.0 - self.ssa)
        return self._abs

    @property
    def ext(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Extinction coefficient array, cached to minimize overhead.
        """
        if self._ext is None:
            self._ext = to_quantity(self.data["ext"])
        return self._ext

    @property
    def scat(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Scattering coefficient array, cached to minimize overhead.
        """
        if self._scat is None:
            self._scat = self.ext * self.ssa
        return self._scat

    @property
    def ssa(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Single-scattering albedo array, cached to minimize overhead.
        """
        if self._ssa is None:
            self._ssa = self.data["ssa"].values * ureg.dimensionless
        return self._ssa

    @property
    def phase(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray.DataArray
            The ``phase`` variable of ``data``, transposed to ensure a correct
            layout during interpolation, cached to minimize overhead.
        """
        if self._phase is None:
            self._phase = self.data["phase"].transpose("phamat", "iangle", "w")
        return self._phase

    @property
    def pmom(self) -> xr.DataArray | None:
        """
        Returns
        -------
        xarray.DataArray
            The ``pmom`` variable of ``data``, cached to minimize overhead.
        """
        if self._pmom is None:
            if "pmom" not in self.data:
                return None
            else:
                self._pmom = self.data["pmom"].transpose("imom", "w")

        return self._pmom

    @property
    def particle_shape(self) -> str:
        """
        Returns
        -------
        str
            Particle shape, described as a string, inferred from dataset contents
            and cached to minimize overhead.
        """
        if self._particle_shape is None:
            if self.data.sizes["phamat"] in {1, 4}:
                self._particle_shape = "spherical"
            elif self.data.sizes["phamat"] == 6:
                self._particle_shape = "spheroidal"
            else:
                raise ValueError("could not guess shape from data")

        return self._particle_shape

    @property
    def has_polarization(self) -> bool:
        """
        Returns
        -------
        bool
            ``True`` iff the phase matrix has more than one coefficient.
        """
        return self.data.sizes["phamat"] > 1

    @property
    def has_fixed_mu_grid(self) -> bool:
        """
        Returns
        -------
        bool
            ``True`` iff all wavelengths share an identical mu grid (same
            ``nangles`` and same ``mu`` values). When ``True``, ``eval_phase``
            uses ``n_iangle`` output points instead of ``2 * n_iangle``,
            preserving the original grid exactly.
        """
        if self._has_fixed_mu_grid is None:
            ds = self.data
            nangles = ds["nangles"].values
            n_iangle = ds.sizes["iangle"]
            all_full = bool(np.all(nangles == n_iangle))
            all_same = all_full and bool(
                np.all(ds["mu"].values == ds["mu"].values[[0]])
            )
            self._has_fixed_mu_grid = all_same

        return self._has_fixed_mu_grid

    def eval_ext(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate the extinction coefficient at wavelength(s) ``w`` by plain
        linear interpolation.

        Parameters
        ----------
        w : quantity
            Query wavelength(s); scalar or array.

        Returns
        -------
        quantity
            Extinction coefficient, same shape as ``w``.
        """
        return np.interp(w, self.w, self.ext)

    def eval_ssa(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate the single-scattering albedo at wavelength ``w``.

        Parameters
        ----------
        w : quantity
            Query wavelength(s); scalar or array.

        Returns
        -------
        quantity
            Dimensionless SSA, same shape as ``w``.

        Notes
        -----
        The spectral interpolation scheme weighs bracketing wavelengths with the
        extinction coefficient, similar to libRadtran. The interpolated SSA is
        computed as::

            ssa(w) = [(1-t)*ext_l*ssa_l + t*ext_r*ssa_r] / ext(w)

        where ``ext(w)`` is the linearly interpolated extinction coefficient.
        When ``ext(w)`` is zero the method falls back to plain linear
        interpolation.
        """
        idx_l, idx_r, t = self._locate(w)
        ext = self.ext.m
        ssa = self.ssa.m
        ext_l, ext_r = ext[idx_l], ext[idx_r]
        ssa_l, ssa_r = ssa[idx_l], ssa[idx_r]
        ext_interp = (1.0 - t) * ext_l + t * ext_r
        linear = (1.0 - t) * ssa_l + t * ssa_r
        weighted = ((1.0 - t) * ext_l * ssa_l + t * ext_r * ssa_r) / np.where(
            ext_interp == 0.0, 1.0, ext_interp
        )
        ssa_interp = np.where(ext_interp == 0.0, linear, weighted)
        return ssa_interp * ureg.dimensionless

    def _get_mu_phase(self, w_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the valid mu grid and phase values for a single wavelength index.

        Strips NaN-padding using ``nangles[w_idx]``.

        Parameters
        ----------
        w_idx : int
            Index along the ``w`` dimension.

        Returns
        -------
        mu : ndarray
            Scattering angle cosines, shape ``(nangles[w_idx],)``.

        phase : ndarray
            Phase values, shape ``(n_phamat, nangles[w_idx])``.
        """
        ds = self.data
        nangles = int(ds["nangles"].values[w_idx])
        mu = ds["mu"].values[w_idx, :nangles]
        # phase is cached as (phamat, iangle, w)
        phase = self.phase.values[:, :nangles, w_idx]  # (n_phamat, nangles)
        return mu, phase

    def _locate(self, w: pint.Quantity) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Locate the bracketing wavelength indices and linear weight for ``w``.

        Parameters
        ----------
        w : quantity
            Query wavelength(s); scalar or array.

        Returns
        -------
        idx_l : ndarray
            Indices of the left bracketing wavelength, shape ``(nw,)``.

        idx_r : ndarray
            Indices of the right bracketing wavelength, shape ``(nw,)``.

        t : ndarray
            Linear interpolation weight for ``idx_r`` (0 at ``idx_l``, 1 at
            ``idx_r``), shape ``(nw,)``.
        """
        w_m = np.atleast_1d(w.to(ucc.get("wavelength")).m)
        w_arr = self.w.m

        if len(w_arr) == 1:
            # Single spectral point: nearest-neighbour, no interpolation
            nw = w_m.size
            zero = np.zeros(nw, dtype=int)
            return zero, zero, np.zeros(nw)

        idx_r = np.clip(np.searchsorted(w_arr, w_m, side="right"), 1, len(w_arr) - 1)
        idx_l = idx_r - 1
        w_l, w_r = w_arr[idx_l], w_arr[idx_r]
        t = np.where(w_l == w_r, 0.0, (w_m - w_l) / (w_r - w_l))
        return idx_l, idx_r, t

    def _bracket_and_weights(
        self, w: pint.Quantity
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Locate the bracketing wavelength indices for ``w`` and compute
        scattering-coefficient-weighted interpolation weights for the phase
        function, matching libRadtran's cascaded extinction- then SSA-weighting.

        Parameters
        ----------
        w : quantity
            Query wavelength(s); scalar or array.

        Returns
        -------
        idx_l : ndarray
            Indices of the left bracketing wavelength, shape ``(nw,)``.

        idx_r : ndarray
            Indices of the right bracketing wavelength, shape ``(nw,)``.

        w0 : ndarray
            Weights for ``idx_l``, shape ``(nw,)``.

        w1 : ndarray
            Weights for ``idx_r``, shape ``(nw,)``.

        scat_denom : ndarray
            Linearly interpolated scattering coefficient, shape ``(nw,)``.
            Zero where there is no scattering at either bracketing wavelength.

        Notes
        -----
        Weights are proportional to the scattering coefficient (extinction ×
        SSA) at each bracket wavelength, which is equivalent to libRadtran's
        cascaded scheme where extinction-weighted SSA interpolation is followed
        by SSA-weighted phase interpolation. Where the scattering coefficient is
        zero at both brackets, the weights fall back to plain linear
        interpolation in wavelength.
        """
        idx_l, idx_r, t = self._locate(w)

        scat_arr = self.scat.m
        scat_l = scat_arr[idx_l]
        scat_r = scat_arr[idx_r]
        scat_denom = (1.0 - t) * scat_l + t * scat_r

        safe_denom = np.where(scat_denom == 0.0, 1.0, scat_denom)
        w0 = np.where(scat_denom == 0.0, 1.0 - t, (1.0 - t) * scat_l / safe_denom)
        w1 = np.where(scat_denom == 0.0, t, t * scat_r / safe_denom)

        return idx_l, idx_r, w0, w1, scat_denom

    def eval_phase(
        self, w: pint.Quantity, n_mu: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the phase function at wavelength ``w`` by interpolation.

        Parameters
        ----------
        w : quantity
            Query wavelength (scalar only; array input raises ``ValueError``
            because each wavelength can have a different angular grid).

        n_mu : int, optional
            Number of output mu points. Defaults to ``2 * n_iangle``, where
            ``n_iangle = data.sizes["iangle"]``. Since each bracketing
            wavelength has at most ``n_iangle`` valid angular points, their
            union is at most ``2 * n_iangle``, so the default never triggers
            RDP decimation in practice.

        Returns
        -------
        mu : ndarray
            Scattering angle cosines of the output grid, shape ``(n_mu,)``.

        phase : ndarray
            Phase function values in sr⁻¹, shape ``(n_phamat, n_mu)``.

        Notes
        -----
        The spectral interpolation scheme is similar to libRadtran's and weighs
        bracketing wavelengths by their scattering coefficient.

        When the two bracketing wavelengths have different mu grids, the phase
        function is resampled onto their union grid before interpolation. The
        result is then brought to exactly ``n_mu`` points: decimated with a
        greedy log-space RDP algorithm when the union is larger, or upsampled
        with linear interpolation in μ when it is smaller. Upsampling via linear
        interpolation preserves the piecewise-linear shape that Mitsuba uses
        during tabulated phase-function look-up, so no distortion is introduced.
        """
        if not np.isscalar(w.m):
            raise ValueError("eval_phase() only accepts a scalar wavelength")

        n_iangle = self.data.sizes["iangle"]
        if n_mu is None:
            n_mu = n_iangle if self.has_fixed_mu_grid else 2 * n_iangle

        # --- Step 1: bracket wavelengths and compute mixing weights ---
        idx_l_arr, idx_r_arr, w0_arr, w1_arr, _ = self._bracket_and_weights(w)
        idx_l = int(idx_l_arr[0])
        idx_r = int(idx_r_arr[0])
        w0 = float(w0_arr[0])
        w1 = float(w1_arr[0])

        mu1, phase1 = self._get_mu_phase(idx_l)
        mu2, phase2 = self._get_mu_phase(idx_r)

        # --- Step 2: union mu grid ---
        mu_union = np.union1d(mu1, mu2)

        # --- Step 3: resample both phase functions onto union grid ---
        # interp1d expects (..., n) layout; phase is (n_phamat, nangles[w_idx])
        phase1_union = interp1d(mu1, phase1, mu_union, bounds="clamp")
        phase2_union = interp1d(mu2, phase2, mu_union, bounds="clamp")

        # --- Step 4: scattering-weighted spectral interpolation ---
        phase_union = w0 * phase1_union + w1 * phase2_union

        # --- Step 5: bring to exactly n_mu points ---
        n_union = len(mu_union)

        if n_union > n_mu:
            # Decimate: keep the most informative points in log space
            keep = _rdp1d_log(mu_union, phase_union, n_mu)
            mu_out = mu_union[keep]
            phase_out = phase_union[:, keep]

        elif n_union < n_mu:
            # Upsample: insert midpoints into the largest gaps of the union grid.
            # All original points are preserved, so the non-uniform  density near
            # the forward-scattering peak is maintained.
            mu_out = _upsample_mu(mu_union, n_mu)
            phase_out = interp1d(mu_union, phase_union, mu_out, bounds="clamp")

        else:
            mu_out = mu_union
            phase_out = phase_union

        return mu_out, phase_out

    def eval_pmom(self, w: pint.Quantity, clip: bool = False) -> tuple[np.ndarray, int]:
        """
        Evaluate Legendre moments at wavelength ``w`` by interpolation.

        The two bracketing wavelengths are weighted by their scattering coefficient
        (extinction × single-scattering albedo). This ensures that a wavelength
        with no scattering contributes nothing to the interpolated phase function
        regardless of its spectral proximity. NaN values in the source data are
        treated as zero.

        Parameters
        ----------
        w : quantity
            Query wavelength(s); scalar or array.

        clip : bool, optional
            If ``True``, trailing all-zero slices are removed along the ``imom``
            axis, so ``values.shape[0] == nleg``.

        Returns
        -------
        values : ndarray
            Legendre moment array for the (1,1) phase matrix element, shape
            ``(nmom, nw)`` or ``(nleg, nw)`` when ``clip=True``.

        nleg : int
            Index of the last nonzero moment (+ 1) across all queried
            wavelengths.
        """
        idx_l, idx_r, w0, w1, scat_denom = self._bracket_and_weights(w)

        pmom = self.pmom  # (imom, w_data)
        if pmom is None:
            raise ValueError(
                "ParticleProperties.eval_pmom(): No Legendre moments found in loaded data"
            )

        v_l = np.nan_to_num(pmom.values[:, idx_l])  # (imom, nw)
        v_r = np.nan_to_num(pmom.values[:, idx_r])

        values = w0[np.newaxis, :] * v_l + w1[np.newaxis, :] * v_r  # (imom, nw)

        # zero out wavelengths with no scattering
        values = np.where((scat_denom == 0.0)[np.newaxis, :], 0.0, values)

        # Index of last nonzero moment (across all wavelengths) + 1
        nonzero_rows = np.any(values != 0.0, axis=1)  # (imom,)
        nleg = int(np.flatnonzero(nonzero_rows)[-1] + 1) if nonzero_rows.any() else 0

        return values[:nleg, :] if clip else values, nleg
