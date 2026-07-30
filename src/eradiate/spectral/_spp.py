"""
Sample count (spp) distribution across spectral loop iterations.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .grid import CKDSpectralGrid, MonoSpectralGrid, SpectralGrid
from .response import BandSRF, SpectralResponseFunction, UniformSRF
from ..config._settings import SampleAllocation
from ..quad import Quad
from ..units import to_quantity
from ..units import unit_registry as ureg

__all__ = ["spectral_response_weights", "srf_spp_distribution"]


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
        every iteration) or one value per iteration, *e.g.* the number of
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
    floor_total = int(floor.sum())
    if sample_count < floor_total:
        raise ValueError(
            f"cannot distribute {sample_count} samples across {n} spectral loop "
            f"iterations: the sample count must be at least {floor_total} "
            "(alternatively, use the 'uniform' sample allocation policy)"
        )

    result = floor.copy()
    free = np.ones((n,), dtype=bool)
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


def spectral_response_weights(
    spectral_data: xr.DataArray, srf: SpectralResponseFunction
) -> xr.DataArray:
    r"""
    Compute the weight each spectral bin carries in the band aggregate produced
    by :func:`.pipelines.logic.apply_spectral_response`.

    Parameters
    ----------
    spectral_data : DataArray
        Spectral data the weights are computed for. Only its ``w``,
        ``bin_wmin`` and ``bin_wmax`` coordinates are used.

    srf : .SpectralResponseFunction
        Spectral response function to apply.

    Returns
    -------
    DataArray
        Weights, indexed against the ``w`` coordinate of `spectral_data`. They
        sum to 1 when the SRF support is entirely covered by the data.

    Notes
    -----
    SRF weighting turns a set of per-bin values :math:`x_i` into a single band
    value :math:`\sum_i c_i x_i`. This function computes the coefficients
    :math:`c_i`. They are computed by feeding a unit impulse on each bin (*i.e.*
    considering the :math:`x_i = 1`, all other values set to zero) through the
    resampling and integration steps that are to be applied to the actual data.
    Computed weights are consumed by
    :func:`.pipelines.logic.apply_spectral_response`, allowing different usage
    depending on whether the mean or variance of a variable is processed, and
    by :func:`.srf_spp_distribution` to allocate samples.
    """
    if not {"bin_wmin", "bin_wmax"}.issubset(set(spectral_data.coords.keys())):
        raise ValueError(
            "input data is missing 'bin_wmin' and/or 'bin_wmax' coordinates"
        )

    # Evaluate integral of spectral response function within selected interval
    wmin = to_quantity(spectral_data.coords["bin_wmin"]).min()
    wmax = to_quantity(spectral_data.coords["bin_wmax"]).max()
    srf_int = srf.integrate(wmin, wmax)

    if isinstance(srf, BandSRF):
        srf_w = srf.wavelengths
    elif isinstance(srf, UniformSRF):
        srf_w = np.array([wmin.m_as(ureg.nm), wmax.m_as(ureg.nm)]) * ureg.nm
    else:
        raise TypeError(f"unhandled SRF type '{srf.__class__.__name__}'")

    data_w = to_quantity(spectral_data.coords["w"])

    # Spectral grid is the finest between data and SRF grids
    w_units = data_w.units
    w_m = np.array(sorted(set(data_w.m_as(w_units)) | set(srf_w.m_as(w_units))))

    # One row per spectral bin, each holding a unit impulse on that bin. Only
    # the spectral coordinate is carried over: the others play no part here and
    # would only be resampled for nothing.
    basis = xr.DataArray(
        np.eye(spectral_data.sizes["w"]),
        dims=("bin", "w"),
        coords={"w": spectral_data.coords["w"].values},
    )

    # If data var has length 1 on spectral dimension, directly select
    # the value instead of using interpolation (it's a known scipy issue)
    if spectral_data.sizes["w"] == 1:
        basis_values = basis.isel(w=0, drop=True).expand_dims(w=w_m, axis=1)

    # Otherwise, use nearest neighbour interpolation (we assume that the
    # spectral data is constant over each spectral bin)
    else:
        basis_values = basis.interp(
            w=w_m, method="nearest", kwargs={"fill_value": "extrapolate"}
        )

    srf_values = xr.DataArray(srf.eval(w_m * w_units).magnitude, dims="w")
    weights = (basis_values * srf_values).integrate("w") / srf_int.m_as(w_units)

    # Rows of the identity are the bins of the input grid, in its own order
    return xr.DataArray(
        weights.values, dims="w", coords={"w": spectral_data.coords["w"].values}
    )


def _bin_weights(srf: BandSRF, spectral_grid: CKDSpectralGrid) -> np.ndarray:
    """
    SRF weights of each bin of ``spectral_grid``, as applied by the post-processing
    pipeline.
    """
    coords = {
        name: ("w", q.m_as(ureg.nm), {"units": "nm"})
        for name, q in [
            ("w", spectral_grid.wcenters),
            ("bin_wmin", spectral_grid.wmins),
            ("bin_wmax", spectral_grid.wmaxs),
        ]
    }
    data = xr.DataArray(np.zeros(len(spectral_grid.wcenters)), dims="w", coords=coords)
    return spectral_response_weights(data, srf).values


def srf_spp_distribution(
    target: int,
    srf: SpectralResponseFunction,
    spectral_grid: SpectralGrid,
    ckd_quads: list[Quad] | None = None,
    allocation: SampleAllocation | str | None = None,
) -> np.ndarray:
    """
    Distribute a sample count across the spectral loop iterations driven by
    ``spectral_grid``.

    Parameters
    ----------
    target : int
        Sample count target.

    srf : .SpectralResponseFunction
        Spectral response function of the measure.

    spectral_grid : .SpectralGrid
        Spectral grid driving the spectral loop (already selected against
        ``srf``, *e.g.* via :meth:`.SpectralGrid.select`).

    ckd_quads : list of .Quad, optional
        Quadrature rules for each bin in ``spectral_grid``, in the same
        order as ``spectral_grid.wcenters``. Required if ``spectral_grid``
        is a :class:`.CKDSpectralGrid`.

    allocation : .SampleAllocation or str, optional
        Allocation policy. If unset (the default), the value of the
        ``sample_allocation`` setting (environment variable
        ``ERADIATE_SAMPLE_ALLOCATION``) is used.

    Returns
    -------
    ndarray
        Sample count for each spectral loop iteration, in the order the loop
        visits them: wavelengths of ``spectral_grid`` (mono), or bins of
        ``spectral_grid`` and, within each bin, the g-points of its
        quadrature rule (ckd).

    Raises
    ------
    ValueError
        If ``target`` is too low to give every iteration at least one sample.

    Notes
    -----
    With the ``"weighted"`` policy, ``target`` is the sample count spent on
    each output value of the post-processing pipeline:

    * in mono mode, every wavelength is an output value and gets ``target``
      samples, whatever the SRF;
    * in ckd mode, every bin gets ``target`` samples, split across its
      g-points proportionally to :attr:`.Quad.weights`;
    * in ckd mode with a :class:`.BandSRF`, the band aggregate is the output
      value: ``target`` is split across bins proportionally to the SRF weight
      of each bin (see :func:`spectral_response_weights`), then across
      g-points as above. Every bin gets at least one sample per g-point.

    Counts sum exactly to ``target`` per output value, except that every
    iteration gets at least one sample, which makes the split only
    approximately proportional. Proportional allocation minimizes the variance
    of the aggregate if all iterations have the same per-sample variance;
    when they do not, the optimal (Neyman) allocation also weights by the
    per-iteration standard deviation.

    With the ``"uniform"`` policy, ``target`` applies in full to every
    spectral loop iteration, g-points included (behaviour of Eradiate v1.3
    and earlier).

    Samplers other than ``"independent"`` may round each count up when
    rendering (see :func:`.effective_spp`).
    """
    # Imported here because eradiate.config triggers source directory validation
    from ..config import settings

    if allocation is None:
        # Resolved at call time: the setting may be changed at runtime
        allocation = settings.SAMPLE_ALLOCATION
    uniform = SampleAllocation.convert(allocation) is SampleAllocation.UNIFORM

    if isinstance(spectral_grid, MonoSpectralGrid):
        return np.full(len(spectral_grid.wavelengths), target, dtype=np.int64)

    if not isinstance(spectral_grid, CKDSpectralGrid):
        raise TypeError(
            f"unsupported spectral grid type '{type(spectral_grid).__name__}'"
        )

    if ckd_quads is None:
        raise ValueError(
            "ckd_quads must be specified when spectral_grid is a CKDSpectralGrid"
        )

    ng = [len(quad.nodes) for quad in ckd_quads]

    if uniform:
        return np.full(sum(ng), target, dtype=np.int64)

    if isinstance(srf, BandSRF):
        bin_spp = _allocate(target, _bin_weights(srf, spectral_grid), floor=ng)
    else:
        bin_spp = np.full(len(ckd_quads), target, dtype=np.int64)

    return np.concatenate(
        [_allocate(int(n), quad.weights) for n, quad in zip(bin_spp, ckd_quads)]
    )
