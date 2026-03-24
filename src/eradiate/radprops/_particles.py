from __future__ import annotations

import heapq

import attrs
import numpy as np
import pint
import pinttrs
import xarray as xr
from axsdb.math import interp1d

from ..attrs import define, documented
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


# TODO: If _rdp1d_log becomes a performance bottleneck (e.g. with grids up to
#   10k points called frequently), consider rewriting the heap loop with Numba
#   (@njit(cache=True)). The algorithm maps well to scalar JIT compilation and
#   could yield a 50–100x speedup over pure Python with the same logic.
def _rdp1d_log(mu: np.ndarray, values: np.ndarray, n_out: int) -> np.ndarray:
    """
    Decimate a 1D phase function using Ramer-Douglas-Peucker in log space.

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

    log_values = np.log(np.maximum(values, 1e-300))  # shape (n,) or (nphamat, n)

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


@define
class ParticleProperties:
    data: xr.Dataset = documented(attrs.field())
    _w: pint.Quantity | None = pinttrs.field(
        default=None, units=ucc.deferred("wavelength"), init=False, repr=False
    )
    _phase: xr.DataArray | None = attrs.field(default=None, init=False, repr=False)
    _ext: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    _ssa: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)

    @property
    def w(self) -> pint.Quantity:
        """
        Return the wavelength array as a quantity.
        The array is cached to minimize overhead.
        """
        if self._w is None:
            self._w = to_quantity(self.data["w"]).to(ucc.get("wavelength"))
        return self._w

    @property
    def ext(self) -> pint.Quantity:
        """
        Return the extinction coefficient array as a quantity.
        The array is cached to minimize overhead.
        """
        if self._ext is None:
            self._ext = to_quantity(self.data["ext"])
        return self._ext

    @property
    def ssa(self) -> pint.Quantity:
        """
        Return the single-scattering albedo array as a quantity.
        The array is cached to minimize overhead.
        """
        if self._ssa is None:
            self._ssa = self.data["ssa"].values * ureg.dimensionless
        return self._ssa

    @property
    def phase(self) -> xr.DataArray:
        """
        Return the ``phase`` variable of ``data`` transposed to ensure a correct
        layout during interpolation.
        The DataArray is cached to minimize overhead.
        """
        if self._phase is None:
            self._phase = self.data["phase"].transpose("phamat", "iangle", "w")
        return self._phase

    def eval_ext(self, w: pint.Quantity) -> pint.Quantity:
        return np.interp(w, self.w, self.ext)

    def eval_ssa(self, w: pint.Quantity) -> pint.Quantity:
        return np.interp(w, self.w, self.ssa)

    def _get_mu_phase(self, w_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the valid mu grid and phase values for a single wavelength index.

        Handles both 1D ``mu(iangle)`` (legacy, same grid for all wavelengths)
        and 2D ``mu(iangle, w)`` (per-wavelength grid). Respects ``nangle`` if
        present to strip NaN-padding.

        Parameters
        ----------
        w_idx : int
            Index along the ``w`` dimension.

        Returns
        -------
        mu : ndarray
            Scattering angle cosines, shape ``(nangle,)``.
        phase : ndarray
            Phase values, shape ``(nphamat, nangle)``.
        """
        ds = self.data
        mu_coord = ds["mu"]

        if "nangle" in ds:
            nangle = int(ds["nangle"].values[w_idx])
        else:
            nangle = ds.sizes["iangle"]

        if mu_coord.dims == ("iangle",):
            mu = mu_coord.values[:nangle]
        else:  # (iangle, w)
            mu = mu_coord.values[:nangle, w_idx]

        # phase is cached as (phamat, iangle, w)
        phase = self.phase.values[:, :nangle, w_idx]  # (nphamat, nangle)
        return mu, phase

    def eval_phase(self, w: pint.Quantity) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the phase function at wavelength ``w`` by linear interpolation.

        When the two bracketing wavelengths have different mu grids, the phase
        function is resampled onto their union grid before interpolation, then
        decimated back to a grid of optimal size using a greedy log-space RDP
        algorithm.

        Parameters
        ----------
        w : quantity
            Query wavelength (scalar).

        Returns
        -------
        mu : ndarray
            Scattering angle cosines of the output grid, shape ``(nangle,)``.
        phase : ndarray
            Phase function values in sr⁻¹, shape ``(nphamat, nangle)``.
        """
        w_m = float(np.atleast_1d(w.to(ucc.get("wavelength")).m)[0])
        w_arr = self.w.m  # (nw,)
        iangle_size = self.data.sizes["iangle"]

        # --- Step 1: bracket wavelengths ---
        idx_r = int(np.searchsorted(w_arr, w_m, side="right"))
        idx_r = np.clip(idx_r, 1, len(w_arr) - 1)
        idx_l = idx_r - 1

        mu1, phase1 = self._get_mu_phase(idx_l)
        mu2, phase2 = self._get_mu_phase(idx_r)

        # --- Step 2: union mu grid ---
        mu_union = np.union1d(mu1, mu2)

        # --- Step 3: resample both phase functions onto union grid ---
        # interp1d expects (..., n) layout; phase is (nphamat, nangle)
        phase1_union = interp1d(mu1, phase1, mu_union, bounds="clamp")
        phase2_union = interp1d(mu2, phase2, mu_union, bounds="clamp")

        # --- Step 4: spectral interpolation ---
        w_l, w_r = w_arr[idx_l], w_arr[idx_r]
        if w_l == w_r:
            t = 0.0
        else:
            t = (w_m - w_l) / (w_r - w_l)
        phase_union = phase1_union + t * (phase2_union - phase1_union)

        # --- Step 5: decimate to optimised grid ---
        n_out = min(len(mu_union), iangle_size)
        if n_out < len(mu_union):
            # Use the m11 component (index 0) as the representative curve;
            # _rdp1d_log accepts multiple rows and takes the max error across them
            keep = _rdp1d_log(mu_union, phase_union, n_out)
            mu_out = mu_union[keep]
            phase_out = phase_union[:, keep]
        else:
            mu_out = mu_union
            phase_out = phase_union

        return mu_out, phase_out
