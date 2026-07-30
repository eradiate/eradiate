import numpy as np
import pytest

from eradiate.spectral._spp import _allocate, srf_spp_distribution
from eradiate.spectral.ckd_quad import CKDQuadConfig
from eradiate.spectral.grid import CKDSpectralGrid, MonoSpectralGrid
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF
from eradiate.units import unit_registry as ureg


@pytest.fixture(scope="module")
def band_srf():
    """A Gaussian band SRF centred on 520 nm, spanning several grid points."""
    return BandSRF.gaussian(wl_center=520.0 * ureg.nm, fwhm=15.0 * ureg.nm, pad=True)


class TestAllocate:
    """Tests for the ``_allocate()`` function."""

    @pytest.mark.parametrize(
        "total, weights, floor, expected",
        [
            (9, [1.0, 1.0, 1.0], 1, [3, 3, 3]),
            (100, [1.0, 9.0], 1, [10, 90]),
            (20, [1.0, 2.0, 3.0, 4.0], 1, [2, 4, 6, 8]),
            # Exact ties are broken in favour of the leading iterations
            (10, [1.0, 1.0, 1.0], 1, [4, 3, 3]),
            # Tight budget: everyone is pinned to the floor but one
            (4, [1.0, 2.0, 3.0, 4.0], 1, [1, 1, 1, 1]),
            (5, [1.0, 2.0, 3.0, 4.0], 1, [1, 1, 1, 2]),
            # Zero weight: pinned to the floor, the rest is split as usual
            (101, [0.0, 1.0, 9.0], 1, [1, 10, 90]),
            # Floor clamping: the pinned share is taken from the other bins
            (1000, [1.0, 999.0], 10, [10, 990]),
            (1000, [1.0, 1.0, 998.0], [10, 10, 10], [10, 10, 980]),
            # Cascading clamp: pinning the first iteration lifts the second's
            # share, which is still below its own (larger) floor
            (200, [1.0, 5.0, 94.0], [10, 30, 10], [10, 30, 160]),
            # Exact floor budget: nothing is left to distribute
            (12, [1.0, 2.0, 3.0], [2, 4, 6], [2, 4, 6]),
        ],
        ids=[
            "equal_weights",
            "unequal_weights",
            "remainder",
            "tie_break",
            "tight_budget",
            "tight_budget_remainder",
            "zero_weight",
            "floor_scalar",
            "floor_array",
            "floor_cascade",
            "floor_exact",
        ],
    )
    def test_allocation(self, total, weights, floor, expected):
        result = _allocate(total, weights, floor=floor)
        np.testing.assert_array_equal(result, expected)
        assert result.sum() == total

    @pytest.mark.parametrize("seed", range(4))
    def test_invariants(self, seed):
        """
        The sum is exact and the floors are honoured, whatever the weights.
        """
        rng = np.random.default_rng(seed)

        for _ in range(500):
            n = int(rng.integers(1, 10))
            weights = rng.random(n) * rng.integers(1, 100)
            weights[rng.random(n) < 0.1] = 0.0  # Sprinkle zero weights
            if not weights.any():
                continue
            floor = rng.integers(1, 5, n)
            total = int(floor.sum() + rng.integers(0, 500))

            result = _allocate(total, weights, floor=floor)
            assert result.sum() == total
            assert np.all(result >= floor)

    def test_monotonic(self):
        """Larger weights get more samples (absent floor clamping)."""
        rng = np.random.default_rng(0)
        weights = np.sort(rng.random(10))
        result = _allocate(10000, weights)
        assert np.all(np.diff(result) >= 0)

    @pytest.mark.parametrize(
        "total, weights, floor",
        [
            (2, [1.0, 1.0, 1.0], 1),  # Below the scalar floor
            (9, [1.0, 1.0], [2, 8]),  # Below the per-iteration floor total
        ],
        ids=["scalar_floor", "array_floor"],
    )
    def test_raises_below_floor(self, total, weights, floor):
        with pytest.raises(ValueError, match="cannot distribute"):
            _allocate(total, weights, floor=floor)

    @pytest.mark.parametrize(
        "weights, msg",
        [
            ([1.0, -1.0], "positive or zero"),
            ([0.0, 0.0], "at least one weight must be nonzero"),
        ],
        ids=["negative", "all_zero"],
    )
    def test_raises_invalid_weights(self, weights, msg):
        with pytest.raises(ValueError, match=msg):
            _allocate(100, weights)


class TestSrfSppDistributionMono:
    """Tests for ``srf_spp_distribution()`` in mono mode."""

    @pytest.fixture(scope="class")
    def mono_grid(self):
        return MonoSpectralGrid(
            wavelengths=np.array([500.0, 510.0, 520.0, 530.0, 540.0]) * ureg.nm
        )

    @pytest.mark.parametrize(
        "srf, expected_wavelengths",
        [
            (DeltaSRF(wavelengths=[500.0, 520.0] * ureg.nm), {500.0, 520.0}),
            (
                UniformSRF(wmin=505.0 * ureg.nm, wmax=535.0 * ureg.nm),
                {510.0, 520.0, 530.0},
            ),
        ],
        ids=["delta", "uniform"],
    )
    def test_no_split(self, mode_mono, mono_grid, srf, expected_wavelengths):
        """Delta and uniform SRFs apply the full target to every wavelength."""
        result = srf_spp_distribution(1000, srf, mono_grid.select(srf))
        assert set(result) == expected_wavelengths
        assert all(v == 1000 for v in result.values())

    def test_band_sums_to_target(self, mode_mono, mono_grid, band_srf):
        result = srf_spp_distribution(1000, band_srf, mono_grid.select(band_srf))
        assert sum(result.values()) == 1000
        # More weight (closer to the band center) should get more samples
        assert result[520.0] > result[510.0]
        assert result[520.0] > result[530.0]

    def test_band_below_floor_raises(self, mode_mono, mono_grid, band_srf):
        sel = mono_grid.select(band_srf)
        n = len(sel.wavelengths)
        with pytest.raises(ValueError, match="cannot distribute"):
            srf_spp_distribution(n - 1, band_srf, sel)

    def test_raises_unsupported_srf(self, mode_mono, mono_grid):
        with pytest.raises(TypeError, match="unsupported SRF type"):
            srf_spp_distribution(1000, object(), mono_grid)


class TestSrfSppDistributionCKD:
    """Tests for ``srf_spp_distribution()`` in ckd mode."""

    @pytest.fixture(scope="class")
    def ckd_grid(self):
        return CKDSpectralGrid.arange(
            start=500.0 * ureg.nm, stop=545.0 * ureg.nm, step=10.0 * ureg.nm
        )

    @pytest.fixture(scope="class")
    def ckd_quad_config(self):
        return CKDQuadConfig(type="gauss_legendre", ng_max=4, policy="fixed")

    @staticmethod
    def _quads_for(grid, quad_config):
        return [x[1] for x in grid.walk_quads(quad_config)]

    @pytest.mark.parametrize(
        "srf",
        [
            DeltaSRF(wavelengths=[505.0, 525.0] * ureg.nm),
            UniformSRF(wmin=505.0 * ureg.nm, wmax=535.0 * ureg.nm),
        ],
        ids=["delta", "uniform"],
    )
    def test_no_split_across_bins(self, mode_ckd, ckd_grid, ckd_quad_config, srf):
        sel = ckd_grid.select(srf)
        quads = self._quads_for(sel, ckd_quad_config)
        result = srf_spp_distribution(1000, srf, sel, ckd_quads=quads)

        # Every selected bin gets the full (unsplit) target...
        bins = sorted({w for w, _ in result})
        for w in bins:
            bin_total = sum(v for (bw, _), v in result.items() if bw == w)
            assert bin_total == 1000

        # ...but is itself split across its g-points, weighted by quadrature
        # weight (not uniformly).
        for w in bins:
            g_values = sorted(v for (bw, _), v in result.items() if bw == w)
            assert len(g_values) == ckd_quad_config.ng_max
            assert len(set(g_values)) > 1  # Gauss-Legendre weights are not equal

    def test_band_sums_to_target(self, mode_ckd, ckd_grid, ckd_quad_config, band_srf):
        sel = ckd_grid.select(band_srf)
        quads = self._quads_for(sel, ckd_quad_config)
        target = ckd_quad_config.ng_max * len(sel.wcenters) * 10
        result = srf_spp_distribution(target, band_srf, sel, ckd_quads=quads)

        assert sum(result.values()) == target

        per_bin = {}
        for (w, _g), v in result.items():
            per_bin[w] = per_bin.get(w, 0) + v

        # The bin closest to the band center should get the largest share
        assert per_bin[520.0] == max(per_bin.values())

    def test_band_below_floor_raises(
        self, mode_ckd, ckd_grid, ckd_quad_config, band_srf
    ):
        sel = ckd_grid.select(band_srf)
        quads = self._quads_for(sel, ckd_quad_config)
        min_target = ckd_quad_config.ng_max * len(sel.wcenters)

        with pytest.raises(ValueError, match="cannot distribute"):
            srf_spp_distribution(band_srf, sel, min_target - 1, ckd_quads=quads)

        # But the exact minimum works
        result = srf_spp_distribution(band_srf, sel, min_target, ckd_quads=quads)
        assert sum(result.values()) == min_target

    def test_requires_quads(self, mode_ckd, ckd_grid):
        srf = DeltaSRF(wavelengths=[505.0] * ureg.nm)
        sel = ckd_grid.select(srf)
        with pytest.raises(ValueError, match="ckd_quads must be specified"):
            srf_spp_distribution(1000, srf, sel)

    def test_raises_unsupported_srf(self, mode_ckd, ckd_grid, ckd_quad_config):
        quads = self._quads_for(ckd_grid, ckd_quad_config)
        with pytest.raises(TypeError, match="unsupported SRF type"):
            srf_spp_distribution(1000, object(), ckd_grid, ckd_quads=quads)

    def test_raises_unsupported_grid(self, mode_mono):
        with pytest.raises(TypeError, match="unsupported spectral grid type"):
            srf_spp_distribution(
                1000, DeltaSRF(wavelengths=[500.0] * ureg.nm), object()
            )
