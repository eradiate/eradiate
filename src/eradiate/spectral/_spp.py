"""
Sample count (spp) distribution across spectral loop iterations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .grid import CKDSpectralGrid, MonoSpectralGrid, SpectralGrid
from .response import BandSRF, DeltaSRF, SpectralResponseFunction, UniformSRF
from ..quad import Quad
from ..units import unit_registry as ureg

__all__ = ["srf_spp_distribution"]


def _allocate(
    sample_count: int, weights: ArrayLike, floor: int | ArrayLike = 1
) -> np.ndarray:
    """
    Split an integer sample count across a set of iterations, proportionally
    to ``weights``.

    Every iteration is guaranteed at least ``floor`` samples and the result
    always sums exactly to ``sample_count``.

    Parameters
    ----------
    sample_count : int
        Total sample budget.

    weights : array-like
            Non-negative relative weights, one per iteration.

    floor : int or array-like, default: 1
        Minimum sample count guaranteed to each iteration. Scalar (applied to
        every iteration) or one value per iteration — *e.g.* the number of
        CKD g-points of the corresponding bin, when the result of this call is
        itself about to be split further downstream.

    Returns
    -------
    ndarray
        Integer sample count per iteration, same length as ``weights``.

    Raises
    ------
    ValueError
        If ``sample_count`` is lower than the sum of ``floor`` (cannot give
        every iteration its guaranteed minimum sample count).
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    if np.any(weights < 0.0):
        raise ValueError("all weights must be positive or zero")
    if np.all(weights <= 0.0):
        raise ValueError("at least one weight must be nonzero")

    n = len(weights)
    floor = np.broadcast_to(np.asarray(floor, dtype=np.int64), (n,))
    floor_total = floor.sum()
    if sample_count < floor_total:
        raise ValueError(
            f"cannot distribute {sample_count} samples across {n} spectral loop "
            f"iterations: a minimum of {list(floor)} samples per iteration "
            f"is required (total >= {floor_total})"
        )

    result = floor.copy()
    free = np.ones((n,), dtype=np.bool)
    budget = int(sample_count)

    # Water-filling: split the whole budget proportionally, pin to their floor
    # the iterations that fall short, then redistribute what is left among the
    # remaining ones. Pinning grows the others' share, which may push new
    # iterations below their own floor, hence the loop. The free set is never
    # emptied: the free shares sum to `budget`, itself no lower than the sum of
    # the free floors, so at least one share clears its floor at every pass.
    while True:
        w = weights[free]
        shares = budget * w / w.sum()
        clamped = shares < floor[free]
        if not clamped.any():
            break
        pinned = np.flatnonzero(free)[clamped]
        budget -= int(floor[pinned].sum())
        free[pinned] = False

    # Round the free shares with the largest remainder method, which minimizes
    # the total absolute deviation from the exact (fractional) allocation.
    # Floors are preserved: surviving shares are no lower than their integer
    # floor, hence neither is their integer part.
    base = np.floor(shares).astype(np.int64)
    leftover = budget - int(base.sum())
    if leftover > 0:
        base[np.argsort(base - shares, kind="stable")[:leftover]] += 1
    result[free] = base

    return result


def _trapezoidal_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-node trapezoidal integration weight (local support width times node
    value), used to turn point samples of a spectral response function into
    relative allocation weights.
    """
    if len(x) == 1:
        return np.array([1.0])

    dx = np.empty_like(x)
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    dx[1:-1] = x[2:] - x[:-2]
    return 0.5 * dx * y


def _check_srf(srf: SpectralResponseFunction) -> None:
    if not isinstance(srf, (BandSRF, DeltaSRF, UniformSRF)):
        raise TypeError(f"unsupported SRF type '{type(srf).__name__}'")


def _mono_distribution(
    target: int,
    srf: SpectralResponseFunction,
    spectral_grid: MonoSpectralGrid,
    uniform: bool,
) -> dict[float, int]:
    _check_srf(srf)
    w_nm = spectral_grid.wavelengths.m_as(ureg.nm)

    if isinstance(srf, BandSRF) and not uniform:
        weights = _trapezoidal_weights(w_nm, srf.eval(spectral_grid.wavelengths).m)
        spp = _allocate(target, weights)
    else:
        spp = np.full(len(w_nm), target, dtype=int)

    return {float(w): int(s) for w, s in zip(w_nm, spp)}


def _ckd_distribution(
    target: int,
    srf: SpectralResponseFunction,
    spectral_grid: CKDSpectralGrid,
    ckd_quads: list[Quad],
    uniform: bool,
) -> dict[tuple[float, float], int]:
    _check_srf(srf)
    w_nm = spectral_grid.wcenters.m_as(ureg.nm)
    n_bins = len(w_nm)

    if uniform:
        # Every spectral loop iteration, g-points included, gets the full target
        return {
            (float(w), float(g)): int(target)
            for w, quad in zip(w_nm, ckd_quads)
            for g in quad.eval_nodes([0.0, 1.0])
        }

    if isinstance(srf, BandSRF):
        bin_weights = np.array(
            [
                srf.integrate(wmin, wmax).m_as(ureg.nm)
                for wmin, wmax in zip(spectral_grid.wmins, spectral_grid.wmaxs)
            ]
        )
        # Every bin must receive at least as many samples as it has
        # quadrature points, since its allocation is split further below.
        ng_per_bin = np.array([len(quad.nodes) for quad in ckd_quads])
        bin_spp = _allocate(target, bin_weights, floor=ng_per_bin)
    else:
        bin_spp = np.full(n_bins, target, dtype=int)

    # Within each bin, split its target across quadrature g-points,
    # weighted by quadrature weight, regardless of SRF type.
    result: dict[tuple[float, float], int] = {}
    for w, quad, bin_target in zip(w_nm, ckd_quads, bin_spp):
        g_values = quad.eval_nodes([0.0, 1.0])
        g_spp = _allocate(int(bin_target), quad.weights)
        for g, s in zip(g_values, g_spp):
            result[(float(w), float(g))] = int(s)

    return result


def srf_spp_distribution(
    target: int,
    srf: SpectralResponseFunction,
    spectral_grid: SpectralGrid,
    ckd_quads: list[Quad] | None = None,
    allocation: str | None = None,
) -> dict[float, int] | dict[tuple[float, float], int]:
    """
    Distribute a sample count budget across the spectral loop iterations
    driven by ``spectral_grid``.

    Parameters
    ----------
    target : int
        Sample count budget.

    srf : .SpectralResponseFunction
        Spectral response function driving the distribution policy (see
        ``allocation``).

    spectral_grid : .SpectralGrid
        Spectral grid driving the spectral loop (already selected against
        ``srf``, *e.g.* via :meth:`.SpectralGrid.select`).

    ckd_quads : list of .Quad, optional
        Quadrature rules for each bin in ``spectral_grid``, in the same
        order as ``spectral_grid.wcenters``. Required if ``spectral_grid``
        is a :class:`.CKDSpectralGrid`.

    allocation : str, optional
        Allocation policy, either ``"weighted"`` or ``"uniform"``. If unset
        (the default), the value of the ``sample_allocation`` setting
        (environment variable ``ERADIATE_SAMPLE_ALLOCATION``) is used.

        With ``"weighted"``, ``target`` is a total budget and the returned
        sample counts sum exactly to it. :class:`.DeltaSRF` and
        :class:`.UniformSRF` apply ``target`` in full to every wavelength
        (mono) or bin (ckd); :class:`.BandSRF` distributes it across
        wavelengths (mono) or bins (ckd), weighted by the local SRF integral.
        In ckd mode, whatever sample count applies to a bin is further split
        across that bin's quadrature g-points, weighted by
        :attr:`.Quad.weights`, regardless of SRF type. Every iteration is
        guaranteed at least one sample, hence the distribution is only
        approximately proportional.

        With ``"uniform"``, ``target`` applies in full to *every* spectral
        loop iteration, g-points included, regardless of SRF type: the
        returned sample counts sum to ``target`` times the number of
        iterations. This is the behaviour of Eradiate v1.3 and earlier.

    Returns
    -------
    dict
        In mono mode, maps wavelength [nm] to sample count. In ckd mode,
        maps ``(bin center wavelength [nm], g)`` to sample count. Keys match
        :attr:`.MonoSpectralIndex.as_hashable` /
        :attr:`.CKDSpectralIndex.as_hashable`.
    """
    # Imported here because eradiate.config triggers source directory validation
    from ..config import SAMPLE_ALLOCATION_POLICIES, settings

    if allocation is None:
        # Resolved at call time: the setting may be changed at runtime
        allocation = settings.SAMPLE_ALLOCATION

    if allocation not in SAMPLE_ALLOCATION_POLICIES:
        raise ValueError(
            f"unsupported sample allocation policy '{allocation}' (expected one "
            f"of {list(SAMPLE_ALLOCATION_POLICIES)})"
        )
    uniform = allocation == "uniform"

    if isinstance(spectral_grid, MonoSpectralGrid):
        return _mono_distribution(target, srf, spectral_grid, uniform)

    elif isinstance(spectral_grid, CKDSpectralGrid):
        if ckd_quads is None:
            raise ValueError(
                "ckd_quads must be specified when spectral_grid is a CKDSpectralGrid"
            )
        return _ckd_distribution(target, srf, spectral_grid, ckd_quads, uniform)

    else:
        raise TypeError(
            f"unsupported spectral grid type '{type(spectral_grid).__name__}'"
        )
