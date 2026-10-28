import numpy as np
import pytest

from eradiate.config import settings
from eradiate.spectral._spp import _allocate, _bin_weights, srf_spp_distribution
from eradiate.spectral.ckd_quad import CKDQuadConfig
from eradiate.spectral.grid import CKDSpectralGrid, MonoSpectralGrid
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF
from eradiate.units import unit_registry as ureg

SRFS = {
    "delta": DeltaSRF(wavelengths=[505.0, 525.0] * ureg.nm),
    "uniform": UniformSRF(wmin=505.0 * ureg.nm, wmax=535.0 * ureg.nm),
    "band": BandSRF.gaussian(wl_center=520.0 * ureg.nm, fwhm=15.0 * ureg.nm, pad=True),
}


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
            "floor",
            "floor_cascade",
            "floor_exact",
        ],
    )
    def test_allocation(self, total, weights, floor, expected):
        np.testing.assert_array_equal(_allocate(total, weights, floor=floor), expected)

    def test_invariants(self):
        """The sum is exact and the floors are honoured, whatever the weights."""
        rng = np.random.default_rng(0)

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

    @pytest.mark.parametrize(
        "weights, match",
        [
            ([1.0, np.nan], "finite"),
            ([1.0, np.inf], "finite"),
            ([1.0, -1.0], "positive or zero"),
            ([0.0, 0.0], "nonzero"),
        ],
    )
    def test_raises_bad_weights(self, weights, match):
        with pytest.raises(ValueError, match=match):
            _allocate(10, weights)

    def test_raises_below_floor(self):
        with pytest.raises(ValueError, match="must be at least 10"):
            _allocate(9, [1.0, 1.0], floor=[2, 8])


@pytest.mark.parametrize("allocation", ["weighted", "uniform"])
@pytest.mark.parametrize("srf", SRFS.values(), ids=SRFS.keys())
def test_mono(srf, allocation):
    """Every wavelength gets the full target, whatever the SRF and policy."""
    grid = MonoSpectralGrid(
        wavelengths=np.array([500.0, 510.0, 520.0, 530.0, 540.0]) * ureg.nm
    ).select(srf)
    result = srf_spp_distribution(1000, srf, grid, allocation=allocation)
    np.testing.assert_array_equal(result, 1000)


@pytest.mark.usefixtures("mode_ckd")
class TestCKD:
    """Tests for ``srf_spp_distribution()`` in ckd mode."""

    @pytest.fixture
    def setup(self, request):
        srf = SRFS[request.param]
        grid = CKDSpectralGrid.arange(
            start=500.0 * ureg.nm, stop=545.0 * ureg.nm, step=10.0 * ureg.nm
        ).select(srf)
        quad_config = CKDQuadConfig(type="gauss_legendre", ng_max=4, policy="fixed")
        quads = [quad for _, quad in grid.walk_quads(quad_config)]
        return srf, grid, quads

    @staticmethod
    def per_bin(result, quads):
        return np.split(result, np.cumsum([len(q.nodes) for q in quads])[:-1])

    @pytest.mark.parametrize("setup", SRFS.keys(), indirect=True)
    def test_uniform(self, setup):
        """Every (bin, g) iteration gets the full target."""
        srf, grid, quads = setup
        result = srf_spp_distribution(
            1000, srf, grid, ckd_quads=quads, allocation="uniform"
        )
        assert len(result) == sum(len(q.nodes) for q in quads)
        np.testing.assert_array_equal(result, 1000)

    @pytest.mark.parametrize("setup", ["delta", "uniform"], indirect=True)
    def test_weighted_per_bin(self, setup):
        """
        Every bin gets the full target, split across g-points by quadrature
        weight.
        """
        srf, grid, quads = setup
        result = srf_spp_distribution(
            1000, srf, grid, ckd_quads=quads, allocation="weighted"
        )
        for spp, quad in zip(self.per_bin(result, quads), quads):
            assert spp.sum() == 1000
            # Larger weight, at least as many samples
            order = np.argsort(quad.weights, kind="stable")
            assert np.all(np.diff(spp[order]) >= 0)

    @pytest.mark.parametrize("setup", ["band"], indirect=True)
    def test_weighted_band(self, setup):
        """The target is a band total, split across bins by pipeline SRF weight."""
        srf, grid, quads = setup
        target = 10000
        result = srf_spp_distribution(
            target, srf, grid, ckd_quads=quads, allocation="weighted"
        )
        assert result.sum() == target

        bin_spp = np.array([spp.sum() for spp in self.per_bin(result, quads)])
        weights = _bin_weights(srf, grid)
        np.testing.assert_allclose(
            bin_spp / target, weights / weights.sum(), atol=len(result) / target
        )

    @pytest.mark.parametrize("setup", ["band"], indirect=True)
    def test_weighted_band_floor(self, setup):
        """Every bin needs at least one sample per g-point."""
        srf, grid, quads = setup
        min_target = sum(len(q.nodes) for q in quads)

        with pytest.raises(ValueError, match=f"must be at least {min_target}"):
            srf_spp_distribution(
                min_target - 1, srf, grid, ckd_quads=quads, allocation="weighted"
            )
        result = srf_spp_distribution(
            min_target, srf, grid, ckd_quads=quads, allocation="weighted"
        )
        np.testing.assert_array_equal(result, 1)

    @pytest.mark.parametrize("setup", ["band"], indirect=True)
    def test_setting(self, setup, monkeypatch):
        """An unset ``allocation`` follows the setting, resolved at call time."""
        srf, grid, quads = setup
        monkeypatch.setattr(settings, "SAMPLE_ALLOCATION", "uniform")
        result = srf_spp_distribution(1000, srf, grid, ckd_quads=quads)
        np.testing.assert_array_equal(result, 1000)

        monkeypatch.setattr(settings, "SAMPLE_ALLOCATION", "weighted")
        result = srf_spp_distribution(1000, srf, grid, ckd_quads=quads)
        assert result.sum() == 1000
