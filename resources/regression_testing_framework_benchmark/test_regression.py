import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from robot.api import logger

import eradiate
import eradiate.test_tools.regression as tt
from eradiate.test_tools.test_cases.rami4atm import (
    create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp,
)
from eradiate.test_tools.util import append_doc

test_types = {
    "t_test": tt.IndependentStudentTTest,
    "paired_t_test": tt.PairedStudentTTest,
    "z_test": tt.ZTest,
    "chi2": tt.Chi2Test,
    "sidak_t_test": tt.SidakTTest,
}


def _generate_antithetic_normal(mean=0, std=1, n=11):
    """
    Generate antithetic normal distribution to ensure the mean of the samples
    is close to the theoretical mean, even with lower sample counts. For testing
    purposes only.
    """

    if n % 2 != 0:
        raise ValueError("n must be even for antithetic pairing.")
    half_n = n // 2
    samples1 = np.random.normal(loc=mean, scale=std, size=half_n)
    samples2 = 2 * mean - samples1
    return np.concatenate([samples1, samples2])


@pytest.mark.parametrize(
    "mean_ref,mean_obs,std_ref,std_obs,manual_obs_outliers,accept, threshold",
    [
        (0.0, 0.0, 1.0, 1.0, None, True, 0.99),  # Basic case, mean is 0
        (5.0, 5.0, 1.0, 1.0, None, True, 0.99),  # Basic case, mean is 5
        (0.0, 1.0, 1.0, 1.0, None, False, 0.05),  # Basic case, mean is different
        (0.0, 5e-3, 1.0, 1.0, None, False, 5e-4),  # Slight bias
        (5.0, 5.0, 1e-4, 1e-4, None, True, 0.99),  # Basic case, low variance
        (5.0, 5.0, 1e-4, 1.5, None, True, 0.99),  # High observation variance
        (
            5.0,
            5.0 - 5e-3,
            1e-4,
            1.5,
            None,
            False,
            1e-3,
        ),  # High observation variance, slight bias
        (5.0, 5.0, 1.0, 1.0, [15.0], False, 1e-5),  # Outlier larger than 2 sigma
    ],
)
@pytest.mark.parametrize(
    "test_type",
    [
        "t_test",
        "paired_t_test",
        "sidak_t_test",
    ],
)
@pytest.mark.parametrize(
    "ref_size,obs_size",
    [
        (1024, 1024),
    ],
)
def test_t_test_static_var(
    mean_ref,
    mean_obs,
    std_ref,
    std_obs,
    threshold,
    accept,
    ref_size,
    obs_size,
    manual_obs_outliers,
    test_type,
    tmp_path,
):
    ref = _generate_antithetic_normal(mean=mean_ref, std=std_ref, n=ref_size)
    obs = _generate_antithetic_normal(mean=mean_obs, std=std_obs, n=obs_size)
    if manual_obs_outliers is not None:
        obs = np.append(obs, manual_obs_outliers)
        ref = np.append(ref, [mean_ref for _ in manual_obs_outliers])

    ref_ds = xr.Dataset(
        coords=dict(index=("index", np.arange(len(ref), dtype=int))),
        data_vars=dict(
            test_variable=("index", ref),
            test_variable_var=("index", np.ones((len(ref),)) * std_ref**2),
        ),
    )

    obs_ds = xr.Dataset(
        coords=dict(index=("index", np.arange(len(obs), dtype=int))),
        data_vars=dict(
            test_variable=("index", obs),
            test_variable_var=("index", np.ones((len(obs),)) * std_obs**2),
        ),
    )

    test = test_types[test_type](
        name=test_type,
        reference=ref_ds,
        value=obs_ds,
        variable="test_variable",
        archive_dir=tmp_path,
        threshold=threshold,
    )

    passed, p_value = test._evaluate(diagnostic_chart=True)

    logger.info(f"p-value: {p_value}; threshold: {threshold}")
    logger.info(f"mean_ref: {mean_ref}; mean_obs: {mean_obs}")
    logger.info(f"std_ref: {std_ref}; std_obs: {std_obs}")

    assert passed == accept


@pytest.mark.parametrize(
    "mean_ref,mean_obs,std_obs,manual_obs_outliers,accept, threshold",
    [
        (0.0, 0.0, 0.1, None, True, 0.01),  # Basic case, mean is 0
        (5.0, 5.0, 1.0, None, True, 0.01),  # Basic case, mean is 5
        (0.0, 5.0, 1.0, None, False, 0.99),  # Basic case, mean is different
        (0.0, 5e-3 - 4, 1.0, None, False, 5e-4),  # Slight bias
        (5.0, 5.0, 1e-2, None, True, 0.99),  # Basic case, low variance
        (5.0, 5.0, 2.0, None, True, 0.99),  # High observation variance
        (
            5.0,
            5.0 - 5e-3,
            1e-4,
            None,
            False,
            1e-4,
        ),  # High observation variance, slight bias
        (
            5.0,
            5.0,
            1.0,
            np.random.normal(20.0, 0.2, 100),
            False,
            1e-5,
        ),  # Clear Outliers
    ],
)
@pytest.mark.parametrize(
    "ref_size,obs_size",
    [
        (4096, 4096),
    ],
)
def test_z_test_static_var(
    mean_ref,
    mean_obs,
    std_obs,
    threshold,
    accept,
    ref_size,
    obs_size,
    manual_obs_outliers,
    tmp_path,
):
    # reference variance is assumed to be small
    ref = _generate_antithetic_normal(mean=mean_ref, std=std_obs / 20, n=ref_size)
    obs = _generate_antithetic_normal(mean=mean_obs, std=std_obs, n=obs_size)
    if manual_obs_outliers is not None:
        obs = np.append(obs, manual_obs_outliers)
        ref = np.append(ref, [mean_ref for _ in manual_obs_outliers])

    ref_ds = xr.Dataset(
        coords=dict(index=("index", np.arange(len(ref), dtype=int))),
        data_vars=dict(
            test_variable=("index", ref),
        ),
    )

    obs_ds = xr.Dataset(
        coords=dict(index=("index", np.arange(len(obs), dtype=int))),
        data_vars=dict(
            test_variable=("index", obs),
            test_variable_var=("index", np.ones((len(obs),)) * std_obs**2),
        ),
    )

    test = tt.ZTest(
        name="z-test",
        reference=ref_ds,
        value=obs_ds,
        variable="test_variable",
        archive_dir=tmp_path,
        threshold=threshold,
    )

    logger.info(f"Expecting Z-test to pass: {accept}", also_console=True)
    passed, p_value = test._evaluate(diagnostic_chart=True)
    logger.info(f"Z-test did pass: {passed}", also_console=True)
    logger.info(f"min p-value: {p_value}; threshold: {threshold}", also_console=True)
    logger.info(f"mean_ref: {mean_ref}; mean_obs: {mean_obs}", also_console=True)
    logger.info(f"std_obs: {std_obs}", also_console=True)

    if passed != accept:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.grid()
        ax1.hist(obs, bins=50)
        ax1.axvline(mean_obs, color="red", linestyle="--")
        ax1.set_title("Observation sample")

        ax2.grid()
        ax2.hist(ref, label="variable", bins=50)
        ax2.axvline(mean_ref, color="red", linestyle="--", label="metric")
        ax2.set_title("Reference sample")

        plt.legend()

        chart = tt.figure_to_html(plt.gcf())
        plt.close()
        logger.info(chart, html=True)

    assert passed == accept


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
@pytest.mark.parametrize(
    "spp",
    [64],
)
@pytest.mark.parametrize(
    "test_type",
    [
        "t_test",  # Still a good sanity check. This test can't be used this way in practice because it tests samples and not means and variances of a set of samples
        "paired_t_test",  # Still a good sanity check.
        "z_test",
        "chi2",  # Still a good sanity check.
        "sidak_t_test",
    ],
)
def test_stats_same_simulation(mode_ckd_double, test_type, spp):
    r"""
    Student's T-test system test for type I error detection
    =======================================================

    This is a system test, which compares a simulation result of the current
    branch to itself with varying SPP. It tests the Student T-test
    implementation against false positives

    Expected behaviour
    ------------------

    This test should demonstrate that the two distributions show no mean bias
    wrt their variance.

    """

    exp1 = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    exp1.integrator.moment = True
    result = eradiate.run(exp1, spp=spp)

    logger.info("Displaying test dataset")
    logger.info(result._repr_html_(), html=True)

    r1np = result.radiance.sum(dim="w").squeeze().values
    s1np = np.sqrt(result.radiance_var.sum(dim="w").squeeze().values)

    plt.grid()
    plt.errorbar(x=result.vza, y=r1np, yerr=3 * s1np, label="test value")
    plt.title("Spectrally summed radiance and associated $3\\sigma$")
    plt.xlabel("vza [degree]")
    plt.ylabel(f"radiance [{result.radiance_srf.attrs['units']}]")
    plt.legend()
    chart = tt.figure_to_html(plt.gcf())
    plt.close()
    logger.info(chart, html=True)

    logger.info("The test should pass, even considering a large significance level")

    test = test_types[test_type](
        name="type_I_error_test",
        value=result,
        reference=result,
        threshold=0.99,
        archive_dir="tests/",
        variable="radiance",
    )

    passed, p_value = test._evaluate()
    logger.info(f"Test passed: {passed}")
    logger.info(f"P-value: {p_value}")

    assert passed


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
@pytest.mark.parametrize(
    "spp1, spp2",
    [(1024, 10000), (4096, 10000)],
)
@pytest.mark.parametrize(
    "test_type",  # Only these two test types implement a set of statistical
    # tests based on the mean and variance of a set of samples (i.e. observation
    # of a set pixels at a given wavelength, expressed in terms of mean and
    # variance)
    [
        # "z_test",
        # "sidak_t_test"
        "t_test",
        "paired_t_test",
        "z_test",
        "chi2",
        "sidak_t_test",
    ],
)
def test_stats_same_setup(mode_ckd_double, test_type, spp1, spp2):
    r"""
    Student's T-test system test for type I error detection
    =======================================================

    This is a system test, which compares two simulation results of the
    current branch with varying SPP. It tests the Student T-test implementation
    against false positives

    Expected behaviour
    ------------------

    This test should demonstrate that the two distributions show no mean bias
    wrt their variance.

    """

    exp1 = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    exp1.integrator.moment = True
    result1 = eradiate.run(exp1, spp=spp1)

    logger.info("Displaying test values dataset")
    logger.info(result1._repr_html_(), html=True)

    exp2 = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    exp2.integrator.moment = True
    result2 = eradiate.run(exp2, spp=spp2)

    logger.info("Displaying test reference dataset")
    logger.info(result2._repr_html_(), html=True)

    r1np = result1.radiance.sum(dim="w").squeeze().values
    r2np = result2.radiance.sum(dim="w").squeeze().values
    s1np = np.sqrt(result1.radiance_var.sum(dim="w").squeeze().values)
    s2np = np.sqrt(result2.radiance_var.sum(dim="w").squeeze().values)

    plt.grid()
    plt.errorbar(x=result1.vza, y=r1np, yerr=3 * s1np, label="test value")
    plt.errorbar(x=result2.vza, y=r2np, yerr=3 * s2np, label="test reference")
    plt.title("Spectrally summed radiance and associated $3\\sigma$")
    plt.xlabel("vza [degree]")
    plt.ylabel(f"radiance [{result1.radiance_srf.attrs['units']}]")
    plt.legend()
    chart = tt.figure_to_html(plt.gcf())
    plt.close()
    logger.info(chart, html=True)

    logger.info("The test should pass, even considering a large significance level")

    test = test_types[test_type](
        name="type_I_error_test",
        value=result1,
        reference=result2,
        threshold=0.5,
        archive_dir="tests/",
        variable="radiance",
    )

    passed, p_value = test._evaluate(diagnostic_chart=True)
    logger.info(f"Test passed: {passed}")

    assert passed


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
@pytest.mark.parametrize("bias", [0.005, 0.01, 0.02, 0.03])
@pytest.mark.parametrize(
    "test_type",  # Only these two test types implement a set of statistical
    # tests based on the mean and variance of a set of samples (i.e. observation
    # of a set pixels at a given wavelength, expressed in terms of mean and
    # variance)
    [
        # "z_test",
        # "sidak_t_test",
        # "t_test"
        "t_test",
        "paired_t_test",
        "z_test",
        "chi2",
        "sidak_t_test",
    ],
)
@pytest.mark.parametrize("spp", [10000])
def test_t_test_biased_simulation(mode_ckd_double, test_type, spp, bias):
    r"""
    Student's T-test system test for type II error detection
    ========================================================

    This is a system test, which compares two simulation results of the
    current branch with varying SPP. It tests the Student T-test implementation
    against false negatives, by introducing a significant bias in one of the
    simulation results.

    Expected behaviour
    ------------------

    This test should demonstrate that the two distributions show a signficant
    mean bias wrt their variance.

    """

    exp1 = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    exp1.integrator.moment = True
    result1 = eradiate.run(exp1, spp=spp)

    exp2 = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    exp2.integrator.moment = True
    result2 = eradiate.run(exp2, spp=spp)

    result2["radiance"] = (
        result2.radiance + (result1.radiance.mean() - result2.radiance.mean())
    ) * (1.0 + bias)

    # The test should not pass, even considering a small significance level
    test = test_types[test_type](
        name="type_II_error_test",
        value=result1,
        reference=result2,
        threshold=0.01,
        archive_dir="tests/",
        variable="radiance",
    )

    passed, p_value = test._evaluate(diagnostic_chart=True)

    assert not passed
