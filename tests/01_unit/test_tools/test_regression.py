import numpy as np
import pytest
import xarray as xr

import eradiate.test_tools.regression as tt


@pytest.mark.parametrize(
    "cls, name",
    [
        (tt.RMSETest, "rmse"),
        (tt.Chi2Test, "chi2"),
        (tt.IndependentStudentTTest, "t-test"),
        (tt.PairedStudentTTest, "t-test"),
        (tt.ZTest, "z-test"),
    ],
    ids=["rmse", "chi2", "t-test", "t-test", "z-test"],
)
def test_instantiate(cls, name):
    # instantiate the test with reasonable defaults
    assert cls(
        name=name,
        archive_dir="tests/",
        value=xr.Dataset(),
        reference=xr.Dataset(),
        threshold=0.05,
        plot=False,
    )


def test_instantiate_fail():
    # assert all arguments except reference are needed
    # only one subclass of RegressionTest (Chi2Test) is tested
    with pytest.raises(TypeError):
        assert tt.Chi2Test(
            archive_dir="tests/",
            value=xr.Dataset(),
            reference=xr.Dataset(),
            threshold=0.05,
            plot=False,
        )

    with pytest.raises(TypeError):
        tt.Chi2Test(
            name="chi2",
            value=xr.Dataset(),
            reference=xr.Dataset(),
            threshold=0.05,
            plot=False,
        )

    with pytest.raises(TypeError):
        tt.Chi2Test(
            name="chi2",
            archive_dir="tests/",
            reference=xr.Dataset(),
            threshold=0.05,
            plot=False,
        )

    with pytest.raises(TypeError):
        tt.Chi2Test(
            name="chi2",
            archive_dir="tests/",
            value=xr.Dataset(),
            reference=xr.Dataset(),
            plot=False,
        )

    assert tt.Chi2Test(
        name="chi2",
        archive_dir="tests/",
        value=xr.Dataset(),
        threshold=0.05,
        plot=False,
    )


def test_reference_converter(tmp_path):
    # test proper handling of missing and unreadable reference

    # file does not exist
    assert (
        tt.Chi2Test(
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

        tt.Chi2Test(
            name="chi2",
            archive_dir="tests/",
            value=xr.Dataset(),
            threshold=0.05,
            reference=tempfile,
            plot=False,
        )

    # wrong data type
    with pytest.raises(ValueError, match="Reference must be provided as a Dataset"):
        tt.Chi2Test(
            name="chi2",
            archive_dir="tests/",
            value=xr.Dataset(),
            threshold=0.05,
            reference=np.zeros(25),
            plot=False,
        )


def test_rmse_evaluate():
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

    test = tt.RMSETest(
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


def test_chi2_evaluate(mode_mono):
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

    test = tt.Chi2Test(
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
