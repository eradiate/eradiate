import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats as spstats
import xarray as xr

from eradiate.test_tools.regression import (
    RegressionTest,
    RMSETest,
    ZTest,
    sidak_family_p_value,
)


def make_offset_datasets(offsets):
    """
    Build a (value, reference) pair over a ``vza`` dimension whose elementwise
    difference is exactly ``offsets``. Decoy data variables are added to both
    datasets to catch a test reading the wrong variable.
    """
    offsets = np.asarray(offsets, dtype=float)
    vza = np.linspace(0.0, 60.0, offsets.size)
    ref_da = xr.DataArray(np.ones(offsets.size), dims="vza", coords={"vza": vza})
    value_da = ref_da + offsets

    ref = xr.Dataset({"brf": ref_da, "stuff": ref_da * 0.2, "wrong": ref_da * 321.0})
    value = xr.Dataset(
        {"brf": value_da, "stuff": value_da * 0.1, "wrong": value_da * 123.0}
    )
    return value, ref


class TestConstruction:
    """
    These tests check constructor argument handling.
    """

    @pytest.mark.parametrize("cls", [RMSETest, ZTest], ids=["rmse", "z-test"])
    def test_instantiate(self, cls):
        # the threshold is the only required argument: a test holds a
        # criterion, not the data it is applied to
        test = cls(0.05)
        assert test.threshold == 0.05
        assert test.variable == "brf_srf"
        assert test.dim is None

    def test_instantiate_fail(self):
        with pytest.raises(TypeError):
            RMSETest()

    def test_abstract_metric(self):
        # a subclass that declares no metric cannot be instantiated: the
        # comparison direction is needed to interpret its threshold
        class Incomplete(RegressionTest):
            def _evaluate(self, result, reference):
                raise NotImplementedError

        with pytest.raises(TypeError, match="Unsupported test type"):
            Incomplete(0.05)


class TestPlot:
    """
    These tests check chart generation for data layouts with extra dimensions.
    """

    @staticmethod
    def make_spectral_datasets(n_w=3, n_x=8, offset=0.1):
        """
        Build a (value, reference) pair with the layout a measure actually
        produces: dimensions ``(w, y_index, x_index)`` with ``vza`` a
        *non-dimension* coordinate over ``(x_index, y_index)``. The elementwise
        difference is ``offset``.
        """
        ref_da = xr.DataArray(
            np.ones((n_w, 1, n_x)),
            dims=("w", "y_index", "x_index"),
            coords={
                "w": np.linspace(440.0, 660.0, n_w),
                "vza": (
                    ("x_index", "y_index"),
                    np.linspace(0.0, 60.0, n_x)[:, np.newaxis],
                ),
            },
        )

        ref = xr.Dataset({"brf": ref_da, "brf_var": ref_da * 0.01})
        value = xr.Dataset({"brf": ref_da + offset, "brf_var": ref_da * 0.01})
        return value, ref

    @pytest.mark.parametrize("cls", [RMSETest, ZTest], ids=["rmse", "z-test"])
    def test_spectral(self, cls):
        # a dataset with a dimension other than the angular one must be
        # charted, with that dimension mapped to colour. The datasets use the
        # production layout, where vza is a coordinate over x_index rather than
        # a dimension of its own.
        n_w = 3
        value, ref = self.make_spectral_datasets(n_w=n_w)

        test = cls(0.05, variable="brf")
        figure, axes = test.plot(value, ref, test.evaluate(value, ref))

        try:
            # one reference line and one result line per wavelength on the
            # comparison panel, one line per wavelength on each difference panel
            assert len(axes[0][0].lines) == 2 * n_w
            assert len(axes[1][0].lines) == n_w
            assert len(axes[1][1].lines) == n_w
        finally:
            plt.close(figure)

    def test_without_outcome(self):
        # charting is also the way a *broken* comparison is diagnosed, so it
        # must work with no verdict to report
        value, ref = self.make_spectral_datasets()
        test = RMSETest(0.05, variable="brf")

        figure, axes = test.plot(value, ref)
        try:
            assert "is not available" in axes[0][1].get_title()
        finally:
            plt.close(figure)

    def test_without_vza(self):
        # a dataset holding only hemispherical quantities carries no vza
        # coordinate at all: the chart falls back to the variable's longest
        # dimension instead of raising
        da = xr.DataArray(
            np.ones((3, 1)),
            dims=("w", "y_index"),
            coords={"w": np.linspace(440.0, 660.0, 3)},
        )
        ref = xr.Dataset({"bhr": da})
        value = xr.Dataset({"bhr": da + 0.01})
        test = RMSETest(0.05, variable="bhr")

        figure, axes = test.plot(value, ref, test.evaluate(value, ref))
        try:
            assert axes[0][0].get_xlabel() == "w"
        finally:
            plt.close(figure)

    def test_noref(self):
        # the reference candidate chart: a single panel, one line per slice
        n_w = 3
        value, _ = self.make_spectral_datasets(n_w=n_w)
        test = RMSETest(0.05, variable="brf")

        figure, ax = test.plot_noref(value)
        try:
            assert len(ax.lines) == n_w
        finally:
            plt.close(figure)


class TestSidak:
    """
    These tests check the Šidák family-wise aggregation helper.
    """

    def test_limits(self):
        assert sidak_family_p_value(1.0, 10) == 1.0
        assert sidak_family_p_value(0.0, 10) == 0.0

    def test_small_p(self):
        # the closed form loses all precision for small p; the implementation
        # must not
        p_min, n = 1e-17, 1000
        assert sidak_family_p_value(p_min, n) == pytest.approx(p_min * n, rel=1e-6)


class TestZTest:
    """
    These tests check the Z-test statistic and its Šidák family-wise
    aggregation.
    """

    @staticmethod
    def make_datasets(
        n=100, bias_in_sigma=0.0, var_res=0.5, var_ref=0.5, outlier_in_sigma=None
    ):
        """
        Build a (value, reference) pair whose elementwise difference is
        ``bias_in_sigma`` times the standard error of the difference, i.e.
        sqrt(var_res + var_ref). If ``outlier_in_sigma`` is set, the last
        element is shifted by that many standard errors instead.
        """
        sigma = np.sqrt(var_res + var_ref)
        bias = np.full(n, bias_in_sigma * sigma)
        if outlier_in_sigma is not None:
            bias[-1] = outlier_in_sigma * sigma

        ref = xr.Dataset(
            {
                "brf": xr.DataArray(np.zeros(n), dims="vza"),
                "brf_var": xr.DataArray(np.full(n, var_ref), dims="vza"),
            }
        )
        value = xr.Dataset(
            {
                "brf": xr.DataArray(bias, dims="vza"),
                "brf_var": xr.DataArray(np.full(n, var_res), dims="vza"),
            }
        )
        return value, ref

    @staticmethod
    def family_p_value(p_min, n):
        """Expected reported metric: the Šidák-corrected family-wise p-value."""
        return 1.0 - (1.0 - p_min) ** n

    @pytest.mark.parametrize(
        "bias_in_sigma, expected_passed",
        [(0.0, True), (1.0, True), (5.0, False)],
        ids=["identical", "1-sigma", "5-sigma"],
    )
    def test_sidak_evaluate(self, bias_in_sigma, expected_passed):
        # known-answer check: a uniform k-sigma shift must yield the two-tailed
        # normal p-value 2 * sf(k) for every pair. The standard error uses both
        # variances, so an implementation ignoring the reference variance would
        # report a p-value for k * sqrt(2) instead. The reported metric
        # aggregates the paired p-values with the Šidák correction.
        n = 100
        value, ref = self.make_datasets(n=n, bias_in_sigma=bias_in_sigma)

        outcome = ZTest(0.05, variable="brf").evaluate(value, ref)

        assert outcome.metric_value == pytest.approx(
            self.family_p_value(2.0 * spstats.norm.sf(bias_in_sigma), n)
        )
        assert outcome.passed is expected_passed

    def test_sidak_no_outlier_quota(self):
        # a single 5-sigma outlier among n = 100 pairs must fail the test: the
        # old 99.75% quota tolerated int(0.9975 * 100) = 99 accepted pairs out
        # of 100, plain Šidák tolerates none
        n = 100
        value, ref = self.make_datasets(n=n, outlier_in_sigma=5.0)

        test = ZTest(0.05, variable="brf")
        outcome = test.evaluate(value, ref)

        assert outcome.passed is False
        # the reported metric is comparable with the threshold:
        # passed <=> p > alpha
        assert outcome.metric_value == pytest.approx(
            self.family_p_value(2.0 * spstats.norm.sf(5.0), n)
        )
        assert outcome.metric_value < test.threshold

    def test_missing_reference_variance(self):
        # legacy references carry no variance: fall back to the result variance
        # alone, i.e. the difference is scaled by sqrt(var_res) instead of
        # sqrt(var_res + var_ref), and say so
        var_res = var_ref = 0.5
        value, ref = self.make_datasets(
            bias_in_sigma=1.0, var_res=var_res, var_ref=var_ref
        )
        ref = ref.drop_vars("brf_var")

        outcome = ZTest(0.05, variable="brf").evaluate(value, ref)

        inflated = np.sqrt((var_res + var_ref) / var_res)
        assert outcome.metric_value == pytest.approx(
            self.family_p_value(2.0 * spstats.norm.sf(inflated), value.sizes["vza"])
        )
        assert any("brf_var" in message for message in outcome.warnings)

    def test_requires_result_variance(self):
        # the result variance is mandatory: this is malformed data, not a
        # failed comparison
        value, ref = self.make_datasets()
        value = value.drop_vars("brf_var")

        with pytest.raises(ValueError, match="The result data for this Z-test"):
            ZTest(0.05, variable="brf").evaluate(value, ref)

    @pytest.mark.parametrize(
        "shrink",
        [
            lambda value, ref: (value, ref.isel(vza=slice(0, 4))),
            lambda value, ref: (value.assign(brf_var=ref["brf_var"].isel(vza=0)), ref),
            lambda value, ref: (value, ref.assign(brf_var=ref["brf_var"].isel(vza=0))),
        ],
        ids=["reference", "result-variance", "reference-variance"],
    )
    def test_shape_mismatch(self, shrink):
        # mismatched shapes are malformed data, not a failed comparison. This
        # must be a ValueError and never an AssertionError: `--force-regen`
        # keys reference regeneration on the latter, so a broken comparison
        # would overwrite the reference it could not read
        value, ref = shrink(*self.make_datasets())

        with pytest.raises(ValueError, match="do not have the same shape"):
            ZTest(0.05, variable="brf").evaluate(value, ref)

    def test_details_and_diagnostic(self):
        # the outcome carries what the report prints and what the diagnostic
        # panel draws, so that evaluation needs no logger and no instance state
        value, ref = self.make_datasets(n=100)

        outcome = ZTest(0.05, variable="brf").evaluate(value, ref)

        assert set(outcome.details) == {
            "min p-value",
            "max p-value",
            "n accepted",
            "alpha_1",
            "alpha_0",
        }
        assert outcome.diagnostic_data["z"].shape == (100,)
        assert "min p-value" in str(outcome)


class TestRMSETest:
    """
    These tests check the RMSE metric and its comparison direction.
    """

    @pytest.mark.parametrize(
        "offsets, expected_rmse",
        [
            (np.zeros(16), 0.0),
            (np.full(16, 0.25), 0.25),
            ([3.0, 4.0], 12.5**0.5),
        ],
        ids=["identical", "constant-offset", "mixed-offsets"],
    )
    def test_evaluate(self, offsets, expected_rmse):
        # known-answer check: a constant offset d yields an RMSE of exactly
        # |d|, and [3, 4] over two points yields sqrt((9 + 16) / 2). The
        # threshold is picked so that the three cases also pin the comparison
        # direction: RMSETest passes when the metric is *below* the threshold,
        # the opposite of the p-value-based classes.
        threshold = 0.25
        value, ref = make_offset_datasets(offsets)

        outcome = RMSETest(threshold, variable="brf").evaluate(value, ref)

        assert outcome.metric_value == pytest.approx(expected_rmse)
        assert outcome.passed is (expected_rmse <= threshold)

    def test_shape_mismatch(self):
        # mismatched shapes are malformed data, not a failed comparison
        value, ref = make_offset_datasets(np.zeros(8))
        ref = ref.isel(vza=slice(0, 4))

        with pytest.raises(ValueError, match="do not have the same shape"):
            RMSETest(0.25, variable="brf").evaluate(value, ref)


class TestPerSliceEvaluation:
    """
    These tests check the ``dim`` field, which evaluates the criterion
    independently for each slice of a dimension.
    """

    @staticmethod
    def make_datasets(offsets_per_w, n_x=8):
        """
        Build a (value, reference) pair over ``(w, vza)`` whose difference is
        constant within each wavelength and given by ``offsets_per_w``.
        """
        offsets = np.asarray(offsets_per_w, dtype=float)
        ref_da = xr.DataArray(
            np.ones((offsets.size, n_x)),
            dims=("w", "vza"),
            coords={
                "w": np.linspace(440.0, 660.0, offsets.size),
                "vza": np.linspace(0.0, 60.0, n_x),
            },
        )
        value_da = ref_da + offsets[:, np.newaxis]
        return xr.Dataset({"brf": value_da}), xr.Dataset({"brf": ref_da})

    def test_is_stricter_than_flattening(self):
        # the point of `dim`: one bad slice must fail the test even when the
        # RMSE of the flattened arrays stays under the threshold, because the
        # good slices dilute it
        threshold = 0.25
        value, ref = self.make_datasets([0.0, 0.5, 0.0])

        flat = RMSETest(threshold, variable="brf").evaluate(value, ref)
        per_slice = RMSETest(threshold, variable="brf", dim="w").evaluate(value, ref)

        assert flat.metric_value == pytest.approx(0.5 / np.sqrt(3.0))
        assert per_slice.metric_value == pytest.approx(0.5)
        assert per_slice.passed is False

    def test_reports_worst_slice(self):
        # the reported metric and coordinate identify the slice that decided
        # the verdict
        value, ref = self.make_datasets([0.1, 0.4, 0.2])

        outcome = RMSETest(1.0, variable="brf", dim="w").evaluate(value, ref)

        assert outcome.passed is True
        assert outcome.metric_value == pytest.approx(0.4)
        assert outcome.details["worst w"] == pytest.approx(550.0)

    def test_worst_slice_direction(self):
        # for a p-value the worst slice is the *smallest* metric, not the
        # largest: the aggregation must follow the comparison direction
        n = 50
        w = np.array([440.0, 550.0])
        bias = np.array([0.0, 5.0])  # second wavelength is off by 5 sigma

        ref = xr.Dataset(
            {
                "brf": xr.DataArray(
                    np.zeros((2, n)), dims=("w", "vza"), coords={"w": w}
                ),
                "brf_var": xr.DataArray(np.full((2, n), 0.5), dims=("w", "vza")),
            }
        )
        value = xr.Dataset(
            {
                "brf": xr.DataArray(
                    np.repeat(bias[:, np.newaxis], n, axis=1),
                    dims=("w", "vza"),
                    coords={"w": w},
                ),
                "brf_var": xr.DataArray(np.full((2, n), 0.5), dims=("w", "vza")),
            }
        )

        outcome = ZTest(0.05, variable="brf", dim="w").evaluate(value, ref)

        assert outcome.passed is False
        assert outcome.details["worst w"] == pytest.approx(550.0)

    @pytest.mark.parametrize("bias", [0.0, 0.5, 5.0])
    def test_ztest_matches_flattening(self, bias):
        # a p-value threshold is a false-positive rate, not a tolerance:
        # slicing must not make the test stricter, otherwise the rate would
        # grow with the number of slices
        n_w, n_vza = 10, 50
        ref = xr.Dataset(
            {
                "brf": xr.DataArray(
                    np.zeros((n_w, n_vza)),
                    dims=("w", "vza"),
                    coords={"w": np.linspace(440.0, 660.0, n_w)},
                ),
                "brf_var": xr.DataArray(np.full((n_w, n_vza), 0.5), dims=("w", "vza")),
            }
        )
        value = ref.copy(deep=True)
        value["brf"] = value["brf"] + bias

        flat = ZTest(0.05, variable="brf").evaluate(value, ref)
        per_slice = ZTest(0.05, variable="brf", dim="w").evaluate(value, ref)

        assert per_slice.metric_value == pytest.approx(flat.metric_value)
        assert per_slice.passed is flat.passed
        assert per_slice.threshold == flat.threshold

    def test_collects_warnings(self):
        # warnings raised by individual slices must reach the caller, but the
        # slices all degrade the same way: one message, not one per slice
        value, ref = self.make_datasets([0.0, 0.0])
        for ds in (value, ref):
            ds["brf_var"] = xr.full_like(ds["brf"], 1e-4)
        ref = ref.drop_vars("brf_var")

        outcome = ZTest(0.05, variable="brf", dim="w").evaluate(value, ref)

        assert len(outcome.warnings) == 1
        assert "brf_var" in outcome.warnings[0]

    def test_empty_dimension(self):
        value, ref = self.make_datasets([0.1])
        value = value.isel(w=slice(0, 0))
        ref = ref.isel(w=slice(0, 0))

        with pytest.raises(ValueError, match="is empty"):
            RMSETest(0.25, variable="brf", dim="w").evaluate(value, ref)

    @pytest.mark.parametrize(
        "reduce",
        [
            lambda ds: ds.isel(w=0),  # scalar coordinate, no longer a dimension
            lambda ds: ds.drop_vars("w"),  # dimension without an index
        ],
        ids=["scalar", "no-index"],
    )
    @pytest.mark.parametrize("side", ["value", "reference"], ids=["value", "reference"])
    def test_unusable_dimension(self, reduce, side):
        # `dim` must be an indexed dimension of *both* datasets: anything else
        # is malformed data, and must not surface as the TypeError or KeyError
        # xarray would raise downstream
        datasets = dict(zip(["value", "reference"], self.make_datasets([0.1, 0.2])))
        datasets[side] = reduce(datasets[side])

        with pytest.raises(ValueError, match="no indexed dimension 'w'"):
            RMSETest(0.25, variable="brf", dim="w").evaluate(
                datasets["value"], datasets["reference"]
            )
