"""Unit tests for eradiate.radprops._particles."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.data.convert import make_aer_core_v2
from eradiate.radprops._particles import ParticleProperties, _rdp1d_log, _upsample_mu

W_NM = np.array([400.0, 550.0, 700.0])
MU_1D = np.array([-1.0, 0.0, 1.0])
EXT = np.array([0.5, 1.0, 0.8])  # km^-1
SSA = np.array([0.9, 0.8, 0.7])
W_NM_SINGLE = np.array([550.0])
EXT_SINGLE = np.array([1.0])  # km^-1
SSA_SINGLE = np.array([0.8])

# Phase: (phamat=1, nw=3, nangle=3), positive values
_PHASE_DATA = np.array([[[1.2, 0.5, 2.0], [0.8, 0.4, 1.8], [1.5, 0.6, 2.2]]])

# Legendre moments: (nw=3, imom=5)
_PMOM_DATA = np.zeros((3, 5))
_PMOM_DATA[:, 0] = 1.0
_PMOM_DATA[:, 1] = [0.5, 0.6, 0.4]
_PMOM_DATA[:, 2] = [0.1, 0.15, 0.05]


def make_dataset(*, zero_scat_idx: int | None = None) -> xr.Dataset:
    """
    Build a minimal spec-compliant Aer-Core v2 dataset for ``ParticleProperties``.

    Always produces a 2D ``mu(w, iangle)`` coordinate and an ``nangles(w)``
    variable, as required by the Aer-Core v2 specification.

    Parameters
    ----------
    zero_scat_idx
        If not ``None``, set ``ssa`` to 0 at that wavelength index.
    """
    n_w = len(W_NM)

    ext = EXT.copy()
    ssa = SSA.copy()
    if zero_scat_idx is not None:
        ssa[zero_scat_idx] = 0.0

    # mu: (nw, nangle) — same grid replicated across wavelengths
    mu_vals = np.tile(MU_1D, (n_w, 1))
    theta_vals = np.degrees(np.arccos(mu_vals))

    return make_aer_core_v2(
        w=W_NM * ureg.nm,
        phamat=["11"],
        mu=mu_vals * ureg.dimensionless,
        theta=theta_vals * ureg.deg,
        ext=ext * ureg("km^-1"),
        ssa=ssa * ureg.dimensionless,
        phase=_PHASE_DATA.copy() * ureg("1/sr"),
        pmom=_PMOM_DATA.copy(),
    )


def make_single_w_dataset() -> xr.Dataset:
    """Build a minimal single-wavelength dataset in Aer-Core v2 format."""
    # mu shape: (nw=1, nangle=3)
    mu_vals = np.array([MU_1D])
    mu = mu_vals * ureg.dimensionless
    theta = (np.arccos(mu_vals) * ureg.rad).to("deg")

    # phase shape: (nphamat=1, nw=1, nangle=3)
    phase_vals = np.array([[[1.5, 0.4, 2.1]]])
    phase = phase_vals * ureg("1/sr")

    # pmom shape: (w=1, imom=5)
    pmom = np.zeros((1, 5))
    pmom[0, 0] = 1.0
    pmom[0, 1] = 0.6
    pmom[0, 2] = 0.15

    return make_aer_core_v2(
        w=W_NM_SINGLE * ureg.nm,
        phamat=["11"],
        mu=mu,
        theta=theta,
        ext=EXT_SINGLE * ureg("km^-1"),
        ssa=SSA_SINGLE * ureg.dimensionless,
        phase=phase,
        pmom=pmom,
    )


REFF = np.array([5.0, 15.0])  # micron
VEFF = np.array([0.05, 0.15])  # dimensionless


def make_size_distribution_dataset() -> xr.Dataset:
    """
    Build a minimal Prt v1-shaped dataset (``reff``/``veff`` dimensions
    added to the Aer-Core v2 layout) for ``ParticleProperties``.

    ``ext``/``ssa`` take a distinct value at every ``(w, reff, veff)`` grid
    point, so index-based selection can be checked against known values. The
    angular grid is shared across the whole grid: the ragged/union-grid
    mechanics along ``w`` are already covered by other tests, so this dataset
    only needs to exercise ``reff``/``veff`` selection.
    """
    n_w, n_reff, n_veff = len(W_NM), len(REFF), len(VEFF)

    ext = np.zeros((n_w, n_reff, n_veff))
    ssa = np.zeros((n_w, n_reff, n_veff))
    for ireff in range(n_reff):
        for iveff in range(n_veff):
            ext[:, ireff, iveff] = EXT * (ireff + 1) * (iveff + 1)
            ssa[:, ireff, iveff] = SSA * (1.0 - 0.05 * ireff - 0.01 * iveff)

    mu_vals = np.tile(MU_1D, (n_w, n_reff, n_veff, 1))
    theta_vals = np.degrees(np.arccos(mu_vals))
    phase = np.tile(
        _PHASE_DATA[:, :, np.newaxis, np.newaxis, :], (1, 1, n_reff, n_veff, 1)
    )
    nangles = np.full((n_w, n_reff, n_veff), len(MU_1D), dtype=np.int32)

    return xr.Dataset(
        {
            "ext": (["w", "reff", "veff"], ext, {"units": "km^-1"}),
            "ssa": (["w", "reff", "veff"], ssa, {"units": "dimensionless"}),
            "mu": (
                ["w", "reff", "veff", "iangle"],
                mu_vals,
                {"units": "dimensionless"},
            ),
            "theta": (
                ["w", "reff", "veff", "iangle"],
                theta_vals,
                {"units": "degree"},
            ),
            "phase": (
                ["phamat", "w", "reff", "veff", "iangle"],
                phase,
                {"units": "1/sr"},
            ),
            "nangles": (["w", "reff", "veff"], nangles),
        },
        coords={
            "w": ("w", W_NM, {"units": "nm"}),
            "reff": ("reff", REFF, {"units": "micron"}),
            "veff": ("veff", VEFF, {"units": "dimensionless"}),
            "phamat": ("phamat", ["11"]),
        },
    )


def make_size_distribution_dataset_ragged() -> xr.Dataset:
    """
    Prt v1-shaped dataset where the native angular grid differs across the
    ``(w, reff, veff)`` grid, to exercise ``eval_phase_grid()``'s per-point
    raggedness/padding handling. Two wavelengths, two ``reff`` points, one
    ``veff`` point.

    * ``reff=0``: identical, fully-populated 4-point mu grid at both
      wavelengths (fixed grid -> no padding needed for this point).
    * ``reff=1``: only 3 valid points at w=0, 4 at w=1 (ragged -> union
      smaller than the default ``2 * n_iangle``, triggers upsampling).
    """
    n_w, n_reff, n_veff, n_iangle = 2, 2, 1, 4

    mu_full = np.array([-1.0, -0.3, 0.3, 1.0])
    mu_vals = np.full((n_w, n_reff, n_veff, n_iangle), np.nan)
    theta_vals = np.full((n_w, n_reff, n_veff, n_iangle), np.nan)
    phase = np.full((1, n_w, n_reff, n_veff, n_iangle), np.nan)
    nangles = np.full((n_w, n_reff, n_veff), n_iangle, dtype=np.int32)

    for iw in range(n_w):
        mu_vals[iw, 0, 0, :] = mu_full
        theta_vals[iw, 0, 0, :] = np.degrees(np.arccos(mu_full))
        phase[0, iw, 0, 0, :] = 1.0 + 0.1 * iw

    mu_w0 = np.array([-1.0, 0.0, 1.0])
    mu_vals[0, 1, 0, :3] = mu_w0
    theta_vals[0, 1, 0, :3] = np.degrees(np.arccos(mu_w0))
    phase[0, 0, 1, 0, :3] = 1.0
    nangles[0, 1, 0] = 3

    mu_vals[1, 1, 0, :] = mu_full
    theta_vals[1, 1, 0, :] = np.degrees(np.arccos(mu_full))
    phase[0, 1, 1, 0, :] = 1.2
    nangles[1, 1, 0] = 4

    ext = np.ones((n_w, n_reff, n_veff))
    ssa = 0.9 * np.ones((n_w, n_reff, n_veff))

    return xr.Dataset(
        {
            "ext": (["w", "reff", "veff"], ext, {"units": "km^-1"}),
            "ssa": (["w", "reff", "veff"], ssa, {"units": "dimensionless"}),
            "mu": (
                ["w", "reff", "veff", "iangle"],
                mu_vals,
                {"units": "dimensionless"},
            ),
            "theta": (
                ["w", "reff", "veff", "iangle"],
                theta_vals,
                {"units": "degree"},
            ),
            "phase": (
                ["phamat", "w", "reff", "veff", "iangle"],
                phase,
                {"units": "1/sr"},
            ),
            "nangles": (["w", "reff", "veff"], nangles),
        },
        coords={
            "w": ("w", [400.0, 700.0], {"units": "nm"}),
            "reff": ("reff", [5.0, 15.0], {"units": "micron"}),
            "veff": ("veff", [0.1], {"units": "dimensionless"}),
            "phamat": ("phamat", ["11"]),
        },
    )


@pytest.fixture
def pp() -> ParticleProperties:
    return ParticleProperties(data=make_dataset())


class TestRdp1dLog:
    """Tests for the _rdp1d_log() function."""

    @pytest.mark.parametrize("n_out", [3, 5, 10])
    def test_passthrough(self, n_out):
        """n_out >= n returns np.arange(n)."""
        n = 3
        mu = np.linspace(-1, 1, n)
        values = np.ones(n)
        result = _rdp1d_log(mu, values, n_out)
        np.testing.assert_array_equal(result, np.arange(n))

    def test_endpoints_retained(self):
        """Indices 0 and n-1 always in result."""
        n = 20
        mu = np.linspace(-1, 1, n)
        values = np.exp(-(mu**2))
        result = _rdp1d_log(mu, values, 5)
        assert 0 in result
        assert n - 1 in result

    def test_output_count_and_sorted(self):
        """Result has exactly n_out elements and is sorted ascending."""
        n = 20
        mu = np.linspace(-1, 1, n)
        values = np.exp(-(mu**2))
        n_out = 7
        result = _rdp1d_log(mu, values, n_out)
        assert len(result) == n_out
        assert np.all(np.diff(result) > 0)

    def test_peak_selected(self):
        """Peak index of strongly peaked function is retained."""
        n = 50
        mu = np.linspace(-1, 1, n)
        # Gaussian peaked at mu=0 → index 25 is the peak
        values = np.exp(-((mu / 0.1) ** 2)) + 1e-10
        peak_idx = int(np.argmax(values))
        result = _rdp1d_log(mu, values, 10)
        assert peak_idx in result

    def test_multirow_max_error(self):
        """2D values: critical point for any row is retained."""
        n = 30
        mu = np.linspace(-1, 1, n)
        # Row 0: flat. Row 1: strongly peaked at index 10
        row0 = np.ones(n)
        row1 = np.exp(-(((mu - mu[10]) / 0.05) ** 2)) + 1e-10
        values = np.stack([row0, row1])  # (2, n)
        result = _rdp1d_log(mu, values, 8)
        assert 10 in result


class TestUpsampleMu:
    """Tests for the _upsample_mu() function."""

    @pytest.mark.parametrize("n_out", [3, 2, 1])
    def test_passthrough(self, n_out):
        """n_out <= n returns the original array unchanged."""
        mu = np.linspace(-1, 1, 3)
        result = _upsample_mu(mu, n_out)
        np.testing.assert_array_equal(result, mu)

    def test_preserves_original_points(self):
        """All original points appear in the output."""
        mu = np.array([-1.0, -0.5, 0.5, 1.0])
        result = _upsample_mu(mu, 8)
        for val in mu:
            assert val in result

    def test_output_count_and_sorted(self):
        """Output has exactly n_out points and is sorted ascending."""
        mu = np.linspace(-1, 1, 5)
        n_out = 12
        result = _upsample_mu(mu, n_out)
        assert len(result) == n_out
        assert np.all(np.diff(result) > 0)

    def test_largest_gap_split_first(self):
        """Midpoint of the largest gap is inserted first."""
        mu = np.array([-1.0, 0.0, 0.9, 1.0])  # largest gap: [-1, 0]
        result = _upsample_mu(mu, len(mu) + 1)
        assert -0.5 in result

    def test_nonuniform_grid_preserved(self):
        """Dense region stays dense; only sparse gaps receive new points."""
        # Dense near mu=1 (forward peak), sparse elsewhere
        mu = np.concatenate([np.linspace(-1.0, 0.0, 5), np.linspace(0.9, 1.0, 10)])
        n_out = len(mu) + 5
        result = _upsample_mu(mu, n_out)
        # All new points fall in the sparse region (gaps > 0.2), not the dense tail
        new_pts = np.setdiff1d(result, mu)
        assert np.all(new_pts < 0.9)


class TestParticleProperties:
    """Tests for the ParticleProperties class."""

    def test_shape(self, pp):
        assert pp.particle_shape == "spherical"

    def test_has_polarization(self, pp):
        assert pp.has_polarization is False

    def test_has_size_distribution_false_for_aer_core_v2(self, pp):
        assert pp.has_size_distribution is False

    def test_has_size_distribution_true_for_prt_v1(self):
        pp = ParticleProperties(data=make_size_distribution_dataset())
        assert pp.has_size_distribution is True

    class TestEvalPhaseGrid:
        def test_eval_phase_raises_on_prt_v1(self):
            """eval_phase() refuses a dataset with 'reff'/'veff' dimensions."""
            pp = ParticleProperties(data=make_size_distribution_dataset())
            with pytest.raises(ValueError, match="eval_phase_grid"):
                pp.eval_phase(W_NM[0] * ureg.nm)

        def test_raises_on_aer_core_v2(self, pp):
            """eval_phase_grid() refuses a dataset without 'reff'/'veff' dimensions."""
            with pytest.raises(ValueError, match="'reff'/'veff' dimensions"):
                pp.eval_phase_grid(W_NM[0] * ureg.nm)

        def test_raises_on_array_wavelength(self):
            """eval_phase_grid() only accepts a scalar wavelength."""
            pp = ParticleProperties(data=make_size_distribution_dataset())
            with pytest.raises(ValueError, match="scalar wavelength"):
                pp.eval_phase_grid(W_NM * ureg.nm)

        @pytest.mark.parametrize("reff_idx", [0, 1])
        @pytest.mark.parametrize("veff_idx", [0, 1])
        def test_matches_manual_isel(self, reff_idx, veff_idx):
            """Each grid point matches eval_phase() on a manually isel'd dataset."""
            ds = make_size_distribution_dataset()
            pp = ParticleProperties(data=ds)
            w = 475.0 * ureg.nm  # midpoint [400, 550]

            mu_grid, phase_grid, nangles_grid = pp.eval_phase_grid(w)

            single = ParticleProperties(
                data=ds.isel(reff=reff_idx, veff=veff_idx, drop=True)
            )
            mu_expected, phase_expected = single.eval_phase(w)

            n = int(nangles_grid[reff_idx, veff_idx])
            assert n == len(mu_expected)
            np.testing.assert_allclose(
                mu_grid[reff_idx, veff_idx, :n], mu_expected, rtol=1e-12
            )
            np.testing.assert_allclose(
                phase_grid[:, reff_idx, veff_idx, :n], phase_expected, rtol=1e-10
            )
            # Beyond this point's own valid count, output is NaN-padded
            assert np.all(np.isnan(mu_grid[reff_idx, veff_idx, n:]))

        def test_ragged_grid_padding(self):
            """
            Points with a different native angular resolution end up with a
            different valid count, and the shorter one is NaN-padded up to
            the grid-wide max.
            """
            ds = make_size_distribution_dataset_ragged()
            pp = ParticleProperties(data=ds)
            w = 550.0 * ureg.nm  # midpoint [400, 700], t=0.5

            mu_grid, phase_grid, nangles_grid = pp.eval_phase_grid(w)

            assert mu_grid.shape[:2] == (2, 1)
            assert phase_grid.shape[1:3] == (2, 1)
            # reff=0: identical fixed grid at both wavelengths -> no padding
            assert nangles_grid[0, 0] == 4
            # reff=1: ragged native grids (3 vs 4 points) -> upsampled, more points
            assert nangles_grid[1, 0] > nangles_grid[0, 0]

            n_mu_max = mu_grid.shape[-1]
            assert n_mu_max == nangles_grid.max()
            n0 = int(nangles_grid[0, 0])
            assert np.all(np.isnan(mu_grid[0, 0, n0:]))
            assert np.all(np.isfinite(mu_grid[1, 0, :]))

            # Matches a manual isel of the ragged point
            single = ParticleProperties(data=ds.isel(reff=1, veff=0, drop=True))
            mu_expected, phase_expected = single.eval_phase(w)
            np.testing.assert_allclose(mu_grid[1, 0, :], mu_expected, rtol=1e-12)
            np.testing.assert_allclose(
                phase_grid[:, 1, 0, :], phase_expected, rtol=1e-10
            )

    class TestLocate:
        @pytest.mark.parametrize(
            "w_nm, expected_t",
            [
                (400.0, 0.0),  # left endpoint
                (550.0, 0.0),  # middle node (left side of next segment)
                (700.0, 1.0),  # right endpoint
            ],
        )
        def test_at_nodes(self, pp, w_nm, expected_t):
            """t=0 at left node of a segment, t=1 at its right node."""
            w = w_nm * ureg.nm
            _idx_l, _idx_r, t = pp._locate(w)
            assert t.shape == (1,)
            np.testing.assert_allclose(t[0], expected_t, atol=1e-12)

        def test_midpoint(self, pp):
            """t=0.5 at midpoint between adjacent nodes."""
            w = 475.0 * ureg.nm  # midpoint of [400, 550]
            idx_l, idx_r, t = pp._locate(w)
            np.testing.assert_allclose(t[0], 0.5, atol=1e-12)
            assert idx_l[0] == 0
            assert idx_r[0] == 1

        @pytest.mark.parametrize(
            "w_nm, exp_idx_l, exp_idx_r",
            [
                (300.0, 0, 1),  # below min → boundary segment [0, 1], t < 0
                (800.0, 1, 2),  # above max → boundary segment [1, 2], t > 1
            ],
        )
        def test_out_of_range_extrapolates(self, pp, w_nm, exp_idx_l, exp_idx_r):
            """Out-of-range w uses the boundary segment; t extrapolates beyond [0, 1]."""
            w = w_nm * ureg.nm
            idx_l, idx_r, t = pp._locate(w)
            assert idx_l[0] == exp_idx_l
            assert idx_r[0] == exp_idx_r
            if w_nm < W_NM[0]:
                assert t[0] < 0.0
            else:
                assert t[0] > 1.0

        def test_array_input(self, pp):
            """Vectorized input returns arrays of correct shape."""
            w = np.array([400.0, 550.0, 700.0]) * ureg.nm
            idx_l, idx_r, t = pp._locate(w)
            assert idx_l.shape == (3,)
            assert idx_r.shape == (3,)
            assert t.shape == (3,)

    class TestGetMuPhase:
        def test_returns_correct_values(self, pp):
            """Returns correct (mu, phase) shapes and mu values for each w_idx."""
            for w_idx in range(3):
                mu, phase = pp._get_mu_phase(w_idx)
                assert mu.shape == (3,)
                assert phase.shape == (1, 3)
                np.testing.assert_array_equal(mu, MU_1D)

        def test_nangles_strips_padding(self):
            """nangles[iw] is used to strip trailing NaN-padded values."""
            ds = make_dataset()
            # Override nangles so w index 1 uses only 2 points
            ds["nangles"] = xr.DataArray(np.array([3, 2, 3], dtype=np.int32), dims="w")
            pp = ParticleProperties(data=ds)

            mu1, phase1 = pp._get_mu_phase(1)
            assert mu1.shape == (2,)
            assert phase1.shape == (1, 2)

            mu0, _ = pp._get_mu_phase(0)
            assert mu0.shape == (3,)

    class TestBracketAndWeights:
        def test_sum_to_one(self, pp):
            """w0 + w1 == 1 for any query w."""
            for w_nm in [400.0, 475.0, 550.0, 625.0, 700.0]:
                w = w_nm * ureg.nm
                _, _, w0, w1, _ = pp._bracket_and_weights(w)
                np.testing.assert_allclose(w0 + w1, 1.0, atol=1e-12)

        def test_zero_scattering_both(self):
            """Both brackets zero-scat → plain linear weights."""
            ds = make_dataset(zero_scat_idx=0)
            ds["ssa"].values[1] = 0.0
            pp = ParticleProperties(data=ds)
            w = 475.0 * ureg.nm  # midpoint [400, 550]
            _, _, w0, w1, scat_denom = pp._bracket_and_weights(w)
            np.testing.assert_allclose(w0[0], 0.5, atol=1e-12)
            np.testing.assert_allclose(w1[0], 0.5, atol=1e-12)
            assert scat_denom[0] == 0.0

        def test_zero_scattering_one_side(self):
            """One bracket zero-scat → other bracket gets full weight."""
            ds = make_dataset(zero_scat_idx=0)
            pp = ParticleProperties(data=ds)
            w = 450.0 * ureg.nm
            _, _, w0, w1, _ = pp._bracket_and_weights(w)
            assert w0[0] == pytest.approx(0.0, abs=1e-12)
            assert w1[0] == pytest.approx(1.0, abs=1e-12)

        def test_proportional(self, pp):
            """w0/w1 ≈ ((1-t)*scat_l) / (t*scat_r) in the normal case."""
            w = 475.0 * ureg.nm  # midpoint [400, 550], t=0.5
            idx_l, idx_r, w0, w1, _ = pp._bracket_and_weights(w)
            scat = (pp.ext * pp.ssa).m
            scat_l = scat[idx_l[0]]
            scat_r = scat[idx_r[0]]
            t = 0.5
            expected_w0 = (1 - t) * scat_l / ((1 - t) * scat_l + t * scat_r)
            expected_w1 = t * scat_r / ((1 - t) * scat_l + t * scat_r)
            np.testing.assert_allclose(w0[0], expected_w0, rtol=1e-10)
            np.testing.assert_allclose(w1[0], expected_w1, rtol=1e-10)

    class TestEvalExt:
        @pytest.mark.parametrize("idx", [0, 1, 2])
        def test_at_nodes(self, pp, idx):
            """Returns tabulated value exactly at each node."""
            w = W_NM[idx] * ureg.nm
            result = pp.eval_ext(w)
            np.testing.assert_allclose(result.m, EXT[idx], rtol=1e-12)

        def test_interpolated(self, pp):
            """Midpoint returns arithmetic mean of adjacent values."""
            w = 475.0 * ureg.nm  # midpoint [400, 550]
            result = pp.eval_ext(w)
            expected = 0.5 * (EXT[0] + EXT[1])
            np.testing.assert_allclose(result.m, expected, rtol=1e-10)

        def test_preserves_reff_veff_dims(self):
            """On a Prt v1 dataset, result keeps the 'reff'/'veff' dimensions."""
            ds = make_size_distribution_dataset()
            pp = ParticleProperties(data=ds)
            w = np.array([475.0, 625.0]) * ureg.nm  # midpoints of both segments
            result = pp.eval_ext(w)

            assert result.shape == (2, len(REFF), len(VEFF))
            expected = 0.5 * (ds["ext"].values[[0, 1]] + ds["ext"].values[[1, 2]])
            np.testing.assert_allclose(result.m, expected, rtol=1e-10)

    class TestEvalSsa:
        @pytest.mark.parametrize("idx", [0, 1, 2])
        def test_at_nodes(self, pp, idx):
            """Returns tabulated value exactly at each node."""
            w = W_NM[idx] * ureg.nm
            result = pp.eval_ssa(w)
            np.testing.assert_allclose(result.m, SSA[idx], rtol=1e-10)

        def test_weighted_formula(self, pp):
            """Midpoint result matches extinction-weighted formula."""
            w = 475.0 * ureg.nm  # midpoint [400, 550], t=0.5
            result = pp.eval_ssa(w)
            t = 0.5
            ext_l, ext_r = EXT[0], EXT[1]
            ssa_l, ssa_r = SSA[0], SSA[1]
            ext_interp = (1 - t) * ext_l + t * ext_r
            expected = ((1 - t) * ext_l * ssa_l + t * ext_r * ssa_r) / ext_interp
            np.testing.assert_allclose(result.m, expected, rtol=1e-10)

        def test_zero_ext_fallback(self):
            """When ext=0 at query point, falls back to plain linear interpolation."""
            ds = make_dataset()
            ds["ext"].values[0] = 0.0
            ds["ext"].values[1] = 0.0
            pp = ParticleProperties(data=ds)
            w = 475.0 * ureg.nm
            result = pp.eval_ssa(w)
            expected = 0.5 * SSA[0] + 0.5 * SSA[1]
            np.testing.assert_allclose(result.m, expected, rtol=1e-10)

        def test_preserves_reff_veff_dims(self):
            """On a Prt v1 dataset, result keeps the 'reff'/'veff' dimensions,
            with the extinction-weighted formula applied independently at
            each (reff, veff) grid point."""
            ds = make_size_distribution_dataset()
            pp = ParticleProperties(data=ds)
            w = np.array([475.0, 625.0]) * ureg.nm  # midpoints of both segments
            result = pp.eval_ssa(w)

            assert result.shape == (2, len(REFF), len(VEFF))

            t = 0.5
            for iw, (il, ir) in enumerate([(0, 1), (1, 2)]):
                ext_l = ds["ext"].values[il]
                ext_r = ds["ext"].values[ir]
                ssa_l = ds["ssa"].values[il]
                ssa_r = ds["ssa"].values[ir]
                ext_interp = (1 - t) * ext_l + t * ext_r
                expected = ((1 - t) * ext_l * ssa_l + t * ext_r * ssa_r) / ext_interp
                np.testing.assert_allclose(result.m[iw], expected, rtol=1e-10)

    class TestEvalPhase:
        def test_array_matches_scalar_calls(self, pp):
            """Array w input matches independent per-wavelength scalar calls."""
            w = np.array([400.0, 475.0, 550.0]) * ureg.nm
            mu_out, phase_out = pp.eval_phase(w)
            assert mu_out.shape == (3, phase_out.shape[-1])
            assert phase_out.shape[0] == 1
            assert phase_out.shape[1] == 3

            for i, w_scalar in enumerate(w):
                mu_scalar, phase_scalar = pp.eval_phase(w_scalar)
                np.testing.assert_allclose(mu_out[i], mu_scalar)
                np.testing.assert_allclose(phase_out[:, i, :], phase_scalar)

        @pytest.mark.parametrize("w_nm", [400.0, 475.0, 550.0, 700.0])
        def test_fixed_grid_shape(self, pp, w_nm):
            """Fixed-grid dataset always returns n_iangle points."""
            mu_out, phase_out = pp.eval_phase(w_nm * ureg.nm)
            n_iangle = pp.data.sizes["iangle"]
            assert pp.has_fixed_mu_grid
            assert mu_out.shape[0] == n_iangle
            assert phase_out.shape == (1, n_iangle)
            assert np.all(phase_out > 0)

        def test_different_grids(self):
            """Different mu grids → union used; output is upsampled to 2 * n_iangle."""
            ds = make_dataset()
            ds["nangles"].values[0] = 2
            ds["nangles"].values[1] = 3
            pp = ParticleProperties(data=ds)
            w = 475.0 * ureg.nm
            mu_out, phase_out = pp.eval_phase(w)
            n_iangle = ds.sizes["iangle"]
            assert mu_out.shape[0] == n_iangle * 2
            assert phase_out.shape[1] == n_iangle * 2

        def test_upsample_when_union_smaller_than_default(self):
            """When the union grid is smaller than 2 * n_iangle, output is upsampled."""
            # Build a dataset with n_iangle=10 but only a few valid points per w.
            # Union of the two bracketing grids will be < 20, triggering upsampling.
            nw = 2
            iangle_max = 10

            mu_vals = np.full((nw, iangle_max), np.nan)
            theta_vals = np.full((nw, iangle_max), np.nan)
            phase_vals = np.full((1, nw, iangle_max), np.nan)

            # w=0: 3 valid points; phase=1.0 integrates to 2 over [-1, 1]
            mu_w0 = np.array([-1.0, 0.0, 1.0])
            mu_vals[0, :3] = mu_w0
            theta_vals[0, :3] = np.degrees(np.arccos(mu_w0))
            phase_vals[0, 0, :3] = 1.0

            # w=1: 4 valid points (same range, different grid → union has 5 points)
            mu_w1 = np.array([-1.0, -0.5, 0.5, 1.0])
            mu_vals[1, :4] = mu_w1
            theta_vals[1, :4] = np.degrees(np.arccos(mu_w1))
            phase_vals[0, 1, :4] = 1.0

            nangles = np.array([3, 4], dtype=np.int32)
            ds = make_aer_core_v2(
                w=np.array([400.0, 700.0]) * ureg.nm,
                phamat=["11"],
                mu=mu_vals * ureg.dimensionless,
                theta=theta_vals * ureg.deg,
                ext=np.ones(nw) * ureg("1/km"),
                ssa=0.9 * np.ones(nw) * ureg.dimensionless,
                phase=phase_vals * ureg("1/sr"),
                nangles=nangles,
                check="full",
            )
            pp = ParticleProperties(data=ds)
            mu_out, phase_out = pp.eval_phase(550.0 * ureg.nm)
            assert mu_out.shape[0] == iangle_max * 2
            assert phase_out.shape == (1, iangle_max * 2)
            assert np.all(np.isfinite(phase_out))
            assert np.all(np.diff(mu_out) > 0)

        def test_custom_n_mu(self, pp):
            """Explicit n_mu controls the output grid size."""
            n_mu = 7
            mu_out, phase_out = pp.eval_phase(475.0 * ureg.nm, n_mu=n_mu)
            assert mu_out.shape[0] == n_mu
            assert phase_out.shape[1] == n_mu

    class TestEvalPmom:
        @pytest.mark.parametrize(
            "w_input",
            [400.0 * ureg.nm, np.array([400.0, 550.0]) * ureg.nm],
        )
        def test_shape(self, pp, w_input):
            """Output shape is (nmom, nw) for scalar and array w."""
            values, nleg = pp.eval_pmom(w_input)
            nw = np.atleast_1d(w_input.m).size
            assert values.shape == (5, nw)
            assert isinstance(nleg, int)

        def test_clip(self, pp):
            """clip=True removes trailing all-zero slices along imom; nleg == last nonzero + 1."""
            w = np.array([400.0, 550.0, 700.0]) * ureg.nm
            values_full, nleg_full = pp.eval_pmom(w, clip=False)
            values_clip, nleg_clip = pp.eval_pmom(w, clip=True)
            assert nleg_full == nleg_clip
            assert values_clip.shape[0] == nleg_clip
            np.testing.assert_array_equal(values_clip, values_full[:nleg_clip, :])

        def test_nan_as_zero(self):
            """NaN in pmom data is treated as 0 in the result."""
            ds = make_dataset()
            pmom_vals = ds["pmom"].values.copy()
            pmom_vals[1, 3] = np.nan  # w=1, imom=3
            ds["pmom"].values[:] = pmom_vals
            pp = ParticleProperties(data=ds)
            w = 550.0 * ureg.nm
            values, _ = pp.eval_pmom(w)
            assert np.all(np.isfinite(values))

        def test_zero_scattering(self):
            """Wavelength between two zero-scattering nodes → moments all zero."""
            ds = make_dataset()
            ds["ssa"].values[0] = 0.0
            ds["ssa"].values[1] = 0.0
            pp = ParticleProperties(data=ds)
            w = 475.0 * ureg.nm
            values, nleg = pp.eval_pmom(w)
            np.testing.assert_array_equal(values[:, 0], 0.0)
            assert nleg == 0 or values[:, 0].sum() == 0.0

    class TestSingleWavelength:
        """
        Regression tests for the single-spectral-point edge case in
        ParticleProperties.  With only one wavelength, _locate used to produce
        idx_l = -1 (out-of-bounds) due to np.clip(1, 1, 0) returning 1.
        """

        @pytest.fixture
        def pp_single(self) -> ParticleProperties:
            return ParticleProperties(data=make_single_w_dataset())

        @pytest.mark.parametrize("w_nm", [300.0, 550.0, 800.0])
        def test_locate_returns_zero_indices(self, pp_single, w_nm):
            """_locate returns idx_l=0, idx_r=0, t=0 for any query wavelength."""
            idx_l, idx_r, t = pp_single._locate(w_nm * ureg.nm)
            assert idx_l[0] == 0
            assert idx_r[0] == 0
            np.testing.assert_allclose(t[0], 0.0, atol=1e-15)

        def test_eval_ssa_off_grid(self, pp_single):
            """eval_ssa returns the single tabulated value for an off-grid wavelength."""
            result = pp_single.eval_ssa(300.0 * ureg.nm)
            np.testing.assert_allclose(result.m, SSA_SINGLE[0], rtol=1e-12)

        def test_eval_phase_values_match_data(self, pp_single):
            """Fixed-grid single-w dataset: default output reproduces stored values exactly."""
            ds = make_single_w_dataset()
            n_iangle = ds.sizes["iangle"]
            assert pp_single.has_fixed_mu_grid

            # Default is n_iangle for a fixed grid; stored values reproduced exactly
            _mu_out, phase_out = pp_single.eval_phase(300.0 * ureg.nm)
            assert phase_out.shape == (1, n_iangle)
            # phase dims in Aer-Core v2: (phamat, w, iangle)
            stored_phase = ds["phase"].values[0, 0, :]
            np.testing.assert_allclose(phase_out[0], stored_phase, rtol=1e-12)

        def test_eval_pmom_values_match_data(self, pp_single):
            """eval_pmom reproduces stored moments and works off-grid."""
            ds = make_single_w_dataset()
            values, nleg = pp_single.eval_pmom(300.0 * ureg.nm)
            assert values.shape == (5, 1)
            assert nleg > 0
            # pmom dims in Aer-Core v2: (w, imom)
            stored = ds["pmom"].transpose("imom", "w").values
            np.testing.assert_allclose(values, stored, rtol=1e-12)

        def test_eval_pmom_clip(self, pp_single):
            """clip=True works correctly with a single spectral point."""
            values_full, nleg = pp_single.eval_pmom(550.0 * ureg.nm, clip=False)
            values_clip, nleg_clip = pp_single.eval_pmom(550.0 * ureg.nm, clip=True)
            assert nleg == nleg_clip
            assert values_clip.shape[0] == nleg_clip
            np.testing.assert_array_equal(values_clip, values_full[:nleg_clip, :])
