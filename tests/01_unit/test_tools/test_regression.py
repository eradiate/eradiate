import numpy as np
import pytest
import xarray as xr

from eradiate.test_tools.regression import (
    Chi2Test,
    IndependentStudentTTest,
    PairedStudentTTest,
    RMSETest,
    ZTest,
)
from eradiate.test_tools.report import ReportLogger


class TestRegression:
    @pytest.mark.parametrize(
        "cls, name",
        [
            (RMSETest, "rmse"),
            (Chi2Test, "chi2"),
            (IndependentStudentTTest, "independent_t-test"),
            (PairedStudentTTest, "paired_t-test"),
            (ZTest, "z-test"),
        ],
        ids=["rmse", "chi2", "independent_t-test", "paired_t-test", "z-test"],
    )
    def test_instantiate(self, cls, name):
        # instantiate the test with reasonable defaults
        assert cls(
            name=name,
            archive_dir="tests/",
            value=xr.Dataset(),
            reference=xr.Dataset(),
            threshold=0.05,
            plot=False,
        )

    @pytest.mark.parametrize(
        "missing",
        ["none", "name", "archive_dir", "value", "threshold", "plot"],
    )
    def test_instantiate_fail(self, missing):
        # assert all arguments except reference are needed
        # only one subclass of RegressionTest (Chi2Test) is tested
        kwargs = {
            "name": "chi2",
            "archive_dir": "tests/",
            "value": xr.Dataset(),
            "threshold": 0.05,
            "plot": False,
        }

        if missing in kwargs:
            kwargs.pop(missing)
            with pytest.raises(TypeError):
                Chi2Test(**kwargs)

        else:
            Chi2Test(**kwargs)

    def test_reference_converter(self, tmp_path):
        # test proper handling of missing and unreadable reference

        # file does not exist
        assert (
            Chi2Test(
                name="chi2",
                archive_dir="tests/",
                value=xr.Dataset(),
                threshold=0.05,
                reference="./this/file/doesnot.exist",
                plot=False,
            ).reference
            is None
        )

        # wrong file type
        with pytest.raises(
            ValueError,
            match="did not find a match in any of xarray's currently installed IO backends",
        ):
            tempfile = tmp_path / "hello.txt"
            tempfile.write_text("test")

            Chi2Test(
                name="chi2",
                archive_dir="tests/",
                value=xr.Dataset(),
                threshold=0.05,
                reference=tempfile,
                plot=False,
            )

        # wrong data type
        with pytest.raises(ValueError, match="Reference must be provided as a Dataset"):
            Chi2Test(
                name="chi2",
                archive_dir="tests/",
                value=xr.Dataset(),
                threshold=0.05,
                reference=np.zeros(25),
                plot=False,
            )


class TestEvaluate:
    def test_rmse(self):
        # test the computation of the RMSE value from given data.
        # we give the dataset some wrong data fields to ensure the right
        # data is used

        result = np.random.rand(50)
        ref = np.random.rand(50)

        result_da = xr.DataArray(result)
        ref_da = xr.DataArray(ref)

        result_ds = xr.Dataset(
            data_vars={
                "brf": result_da,
                "stuff": result_da * 0.1,
                "wrong": result_da * 123.0,
            }
        )
        ref_ds = xr.Dataset(
            data_vars={"brf": ref_da, "stuff": ref_da * 0.2, "wrong": ref_da * 321.0}
        )

        rmse_ref = np.linalg.norm(result - ref) / np.sqrt(len(ref))

        test = RMSETest(
            name="rmse",
            value=result_ds,
            reference=ref_ds,
            variable="brf",
            archive_dir="tests/",
            threshold=0.05,
            plot=False,
        )

        _, rmse = test._evaluate()

        assert rmse == rmse_ref

    def test_chi2(self, mode_mono):
        # test the computation of the Chi squared value from given data.
        # we give the dataset some wrong data fields to ensure the right
        # data is used

        import mitsuba as mi

        result_np = np.random.rand(50)
        ref_np = np.random.rand(50)

        histo_bins = np.linspace(ref_np.min(), ref_np.max(), 20)
        histo_ref = np.histogram(ref_np, histo_bins)[0]
        histo_res = np.histogram(result_np, histo_bins)[0]

        # sorting both histograms following the ascending frequencies in
        # the reference. Algorithm from:
        # https://stackoverflow.com/questions/9764298/how-to-sort-two-lists-which-reference-each-other-in-the-exact-same-way
        histo_ref_sorted, histo_res_sorted = zip(
            *sorted(zip(histo_ref, histo_res), key=lambda x: x[0])
        )

        from mitsuba.math_py import rlgamma

        chi2val, dof, pooled_in, pooled_out = mi.math.chi2(
            histo_res_sorted, histo_ref_sorted, 5
        )
        p_value_ref = 1.0 - rlgamma(dof / 2.0, chi2val / 2.0)

        result_da = xr.DataArray(result_np)
        ref_da = xr.DataArray(ref_np)

        result_ds = xr.Dataset(
            data_vars={
                "brf": result_da,
                "stuff": result_da * 0.1,
                "wrong": result_da * 123.0,
            }
        )
        ref_ds = xr.Dataset(
            data_vars={"brf": ref_da, "stuff": ref_da * 0.2, "wrong": ref_da * 321.0}
        )

        test = Chi2Test(
            name="chi2",
            value=result_ds,
            reference=ref_ds,
            variable="brf",
            archive_dir="tests/",
            threshold=0.05,
            plot=False,
        )

        _, p_value = test._evaluate()

        assert p_value == p_value_ref


class TestRegressionReport:
    """
    These tests check reporting infrastructure integration in the regression
    testing components.
    """

    class ReportLoggerSpy(ReportLogger):
        """
        Report logger that records messages and HTML fragments for assertions
        while forwarding them to the active report backend. Content sent
        through this spy therefore shows up in the generated test report and
        can be inspected visually.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.messages = []
            self.fragments = []

        def info(self, msg):
            self.messages.append(msg)
            super().info(msg)

        def html(self, fragment):
            self.fragments.append(fragment)
            super().html(fragment)

    @pytest.fixture
    def spy(self):
        return self.ReportLoggerSpy()

    @staticmethod
    def make_dataset(values, vza):
        return xr.Dataset(
            {"brf": ("vza", values)},
            coords={"vza": ("vza", vza)},
        )

    @pytest.fixture
    def datasets(self):
        vza = np.linspace(-75.0, 75.0, 11)
        ref = np.linspace(0.1, 0.2, 11)
        result = ref + 1e-3
        return self.make_dataset(result, vza), self.make_dataset(ref, vza)

    def test_messages(self, tmp_path, datasets, spy):
        """
        A regression test sends its diagnostic messages to the injected
        report logger and archives result and reference datasets.
        """
        result, ref = datasets

        test = RMSETest(
            name="rmse-pass",
            value=result,
            reference=ref,
            variable="brf",
            threshold=1.0,
            archive_dir=tmp_path,
            plot=False,
            logger=spy,
        )

        assert test.run()
        assert any("Metric value: rmse" in msg for msg in spy.messages)
        assert not spy.fragments  # No plot requested
        assert (tmp_path / "rmse-pass-result.nc").exists()
        assert (tmp_path / "rmse-pass-ref.nc").exists()

    def test_failure(self, tmp_path, datasets, spy):
        """
        A failing regression test reports through the same channel.
        """
        result, ref = datasets

        test = RMSETest(
            name="rmse-fail",
            value=result,
            reference=ref,
            variable="brf",
            threshold=0.0,
            archive_dir=tmp_path,
            plot=False,
            logger=spy,
        )

        assert not test.run()
        assert any("Test did not pass" in msg for msg in spy.messages)

    def test_plot(self, tmp_path, datasets, spy):
        """
        With plotting enabled, the comparison chart is embedded in the report
        as an SVG fragment and saved to the archive directory as a PNG file
        """
        result, ref = datasets

        test = RMSETest(
            name="rmse-plot",
            value=result,
            reference=ref,
            variable="brf",
            threshold=1.0,
            archive_dir=tmp_path,
            plot=True,
            logger=spy,
        )

        assert test.run()
        assert len(spy.fragments) == 1
        assert spy.fragments[0].startswith("<svg")
        assert (tmp_path / "rmse-plot.png").exists()

    def test_noref(self, tmp_path, datasets, spy):
        """
        Without reference data, the test fails, stores the result as a new
        reference candidate, says so in the report and embeds a plot of the
        reference candidate
        """
        result, _ = datasets

        test = RMSETest(
            name="rmse-noref",
            value=result,
            reference=None,
            variable="brf",
            threshold=1.0,
            archive_dir=tmp_path,
            plot=True,
            logger=spy,
        )

        assert not test.run()
        assert any("No reference data found" in msg for msg in spy.messages)
        assert (tmp_path / "rmse-noref-ref.nc").exists()
        assert len(spy.fragments) == 1
        assert spy.fragments[0].startswith("<svg")
