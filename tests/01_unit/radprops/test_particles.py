"""Unit tests for eradiate.radprops._particles."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops._particles import ParticleProperties, _rdp1d_log

# ---------------------------------------------------------------------------
# Dataset factory helpers
# ---------------------------------------------------------------------------

W_NM = np.array([400.0, 550.0, 700.0])
MU_1D = np.array([-1.0, 0.0, 1.0])
EXT = np.array([0.5, 1.0, 0.8])  # km^-1
SSA = np.array([0.9, 0.8, 0.7])

# Phase: (phamat=1, iangle=3, w=3), positive values
_PHASE_DATA = np.array([[[1.2, 0.8, 1.5], [0.5, 0.4, 0.6], [2.0, 1.8, 2.2]]])

# Legendre moments: (imom=5, phamat=1, w=3)
_PMOM_DATA = np.zeros((5, 1, 3))
_PMOM_DATA[0] = 1.0
_PMOM_DATA[1] = [[0.5, 0.6, 0.4]]
_PMOM_DATA[2] = [[0.1, 0.15, 0.05]]


def make_dataset(
    *, with_nangle: bool = False, mu_2d: bool = False, zero_scat_idx: int | None = None
) -> xr.Dataset:
    """
    Build a minimal dataset for ``ParticleProperties``.

    Parameters
    ----------
    with_nangle
        If ``True``, add an ``nangle(w)`` variable (NaN-padded iangle variant).

    mu_2d
        If ``True``, make ``mu`` have dims ``(w, iangle)`` instead of ``(iangle,)``.

    zero_scat_idx
        If not ``None``, set ``ssa`` to 0 at that wavelength index.
    """
    n_w = len(W_NM)
    n_iangle = len(MU_1D)
    n_phamat = 1

    ext = EXT.copy()
    ssa = SSA.copy()
    if zero_scat_idx is not None:
        ssa[zero_scat_idx] = 0.0

    phase = _PHASE_DATA.copy()  # (1, 3, 3)
    pmom = _PMOM_DATA.copy()  # (5, 1, 3)

    coords: dict = {
        "w": ("w", W_NM, {"units": "nm"}),
        "phamat": ("phamat", np.arange(n_phamat)),
        "imom": ("imom", np.arange(5)),
    }

    if mu_2d:
        # Each wavelength gets its own (identical here) mu row
        mu_vals = np.tile(MU_1D, (n_w, 1))  # (w, iangle)
        coords["mu"] = (("w", "iangle"), mu_vals)
    else:
        coords["mu"] = ("iangle", MU_1D)

    data_vars: dict = {
        "ext": ("w", ext, {"units": "km^-1"}),
        "ssa": ("w", ssa),
        "phase": (("phamat", "iangle", "w"), phase),
        "pmom": (("imom", "phamat", "w"), pmom),
    }

    if with_nangle:
        # All wavelengths use all iangle points (no actual padding here)
        data_vars["nangle"] = ("w", np.full(n_w, n_iangle, dtype=float))

    return xr.Dataset(data_vars=data_vars, coords=coords)


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


class TestParticleProperties:
    """Tests for the ParticleProperties class."""

    def test_constructor(self):
        ds = make_dataset()
        pp = ParticleProperties(data=ds)
        assert pp.data is ds

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
            idx_l, idx_r, t = pp._locate(w)
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
        def test_1d_mu(self, pp):
            """1D mu coord: returns correct (mu, phase) shapes."""
            for w_idx in range(3):
                mu, phase = pp._get_mu_phase(w_idx)
                assert mu.shape == (3,)
                assert phase.shape == (1, 3)

        def test_with_nangle(self):
            """nangle strips NaN-padded trailing values."""
            ds = make_dataset(with_nangle=True)
            # Override nangle so wl index 1 uses only 2 points
            nangle_vals = np.array([3.0, 2.0, 3.0])
            ds["nangle"] = xr.DataArray(nangle_vals, dims="w")
            pp = ParticleProperties(data=ds)

            mu1, phase1 = pp._get_mu_phase(1)
            assert mu1.shape == (2,)
            assert phase1.shape == (1, 2)

            mu0, _ = pp._get_mu_phase(0)
            assert mu0.shape == (3,)

        def test_2d_mu(self):
            """2D mu coord (w, iangle): correct slice per w_idx."""
            ds = make_dataset(mu_2d=True)
            pp = ParticleProperties(data=ds)
            for w_idx in range(3):
                mu, phase = pp._get_mu_phase(w_idx)
                assert mu.shape == (3,)
                np.testing.assert_array_equal(mu, MU_1D)

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

    class TestEvalPhase:
        def test_array_raises(self, pp):
            """Array w input raises ValueError."""
            w = np.array([400.0, 550.0]) * ureg.nm
            with pytest.raises(ValueError, match="scalar"):
                pp.eval_phase(w)

        @pytest.mark.parametrize("idx", [0, 1, 2])
        def test_at_node(self, pp, idx):
            """At exact tabulated wavelength, shapes are correct and phase is positive."""
            w = W_NM[idx] * ureg.nm
            mu_out, phase_out = pp.eval_phase(w)
            assert mu_out.shape[0] <= len(MU_1D)
            assert phase_out.shape == (1, mu_out.shape[0])
            assert np.all(phase_out > 0)

        def test_same_grid(self, pp):
            """Same mu at both brackets → union is same grid, weighted mix."""
            w = 475.0 * ureg.nm
            mu_out, phase_out = pp.eval_phase(w)
            assert mu_out.shape[0] <= len(MU_1D)
            assert phase_out.shape[1] == mu_out.shape[0]
            assert np.all(phase_out > 0)

        def test_different_grids(self):
            """Different mu grids → union used; output nangle <= iangle_size."""
            ds = make_dataset(with_nangle=True)
            ds["nangle"].values[0] = 2.0
            ds["nangle"].values[1] = 3.0
            pp = ParticleProperties(data=ds)
            w = 475.0 * ureg.nm
            mu_out, phase_out = pp.eval_phase(w)
            iangle_size = ds.sizes["iangle"]
            assert mu_out.shape[0] <= iangle_size
            assert phase_out.shape[1] == mu_out.shape[0]

    class TestEvalPmom:
        @pytest.mark.parametrize(
            "w_input",
            [400.0 * ureg.nm, np.array([400.0, 550.0]) * ureg.nm],
        )
        def test_shape(self, pp, w_input):
            """Output shape is (nmom, nphamat, nw) for scalar and array w."""
            values, nleg = pp.eval_pmom(w_input)
            nw = np.atleast_1d(w_input.m).size
            assert values.shape == (5, 1, nw)
            assert isinstance(nleg, int)

        def test_clip(self, pp):
            """clip=True removes trailing all-zero rows; nleg == last nonzero + 1."""
            w = np.array([400.0, 550.0, 700.0]) * ureg.nm
            values_full, nleg_full = pp.eval_pmom(w, clip=False)
            values_clip, nleg_clip = pp.eval_pmom(w, clip=True)
            assert nleg_full == nleg_clip
            assert values_clip.shape[0] == nleg_clip
            np.testing.assert_array_equal(values_clip, values_full[:nleg_clip])

        def test_nan_as_zero(self):
            """NaN in pmom data is treated as 0 in the result."""
            ds = make_dataset()
            pmom_vals = ds["pmom"].values.copy()
            pmom_vals[3, 0, 1] = np.nan
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
            np.testing.assert_array_equal(values[:, :, 0], 0.0)
            assert nleg == 0 or values[:, :, 0].sum() == 0.0
