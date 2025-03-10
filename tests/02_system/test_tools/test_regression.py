import numpy as np
import pytest
import xarray as xr

import eradiate
import eradiate.test_tools.regression as tt
from eradiate.test_tools.test_cases.rami4atm import (
    create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp,
)
from eradiate.test_tools.util import append_doc

test_types = {
    "t_test": tt.IndependantStudentTTest,
    "sidak_t_test": tt.SidakTTest,
}


@pytest.mark.parametrize(
    "mean_ref,mean_obs,var_ref,var_obs,manual_obs_outliers, accept",
    [
        (0.0, 0.0, 1.0, 1.0, None, True),  # Basic case, mean is 0
        (4.0, 4.0, 1.0, 1.0, None, True),  # Mean is the same
        (
            0.0,
            0.0,
            1.0,
            0.5,
            None,
            True,
        ),  # Mean is zero, variances vary between ref and obs
        (0.0, 4.0, 1.0, 1.0, None, False),  # Different mean
        (0.0, 4.0, 2.0, 1.0, None, False),  # Different mean, different variance
        (
            0.0,
            0.01,
            1.0,
            1.0,
            None,
            True,
        ),  # small relative mean variations are discarded
        (
            0.1,
            1.8,
            0.5,
            0.4,
            None,
            False,
        ),  # larger relative mean variations are detected
        (2.0, 2.0, 0.1, 0.2, [-300], False),  # introducing a sharp outlier
        (2.0, 2.0, 0.1, 0.2, [3.0], True),  # introducing a small outlier
    ],
)
@pytest.mark.parametrize(
    "threshold",
    [
        0.05,
        0.01,
        0.002,
        0.0005,
    ],
)
@pytest.mark.parametrize(
    "test_type",
    [
        "t_test",
    ],
)
@pytest.mark.parametrize(
    "ref_size,obs_size,nb_iterations",
    [
        (256, 256, 10000),
    ],
)
def test_t_test_static_var(
    mean_ref,
    mean_obs,
    var_ref,
    var_obs,
    threshold,
    accept,
    ref_size,
    obs_size,
    nb_iterations,
    manual_obs_outliers,
    test_type,
):
    nb_passed = 0
    p_values = []

    for _ in range(nb_iterations):
        ref = np.random.normal(loc=mean_ref, scale=np.sqrt(var_ref), size=ref_size)
        ref_ds = xr.Dataset(
            coords=dict(index=("index", np.arange(ref_size, dtype=int))),
            data_vars=dict(
                test_variable=("index", ref),
                test_variable_var=("index", np.ones((ref_size,)) * var_ref),
            ),
        )

        obs = np.random.normal(loc=mean_obs, scale=np.sqrt(var_obs), size=obs_size)
        if manual_obs_outliers is not None:
            positions = np.random.randint(0, obs_size, len(manual_obs_outliers))
            obs[positions] = manual_obs_outliers
        obs_ds = xr.Dataset(
            coords=dict(index=("index", np.arange(obs_size, dtype=int))),
            data_vars=dict(
                test_variable=("index", obs),
                test_variable_var=("index", np.ones((obs_size,)) * var_obs),
            ),
        )

        try:
            test = test_types[test_type](
                name="t-test",
                reference=ref_ds,
                value=obs_ds,
                variable="test_variable",
                archive_dir="tests/",
                threshold=threshold,
            )
        except TypeError:
            test = test_types[test_type](
                name="t-test",
                reference=ref_ds,
                value=obs_ds,
                variable="test_variable",
                archive_dir="tests/",
                threshold=threshold,
                spo=20000,
            )

        passed, p_value = test._evaluate()
        p_values.append(p_value)

        if passed == accept:
            nb_passed += 1

    assert (
        nb_passed >= nb_iterations * (1 - threshold) * 0.9975
    )  # allow a bit of margin


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
@pytest.mark.slow
@pytest.mark.parametrize(
    "spp1, spp2",
    [
        (16, 16),
        (256, 256),
        (256, 1024),
    ],
)
def test_t_test_same_simulation(mode_ckd_double, spp1, spp2):
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

    exp2 = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    exp2.integrator.moment = True
    result2 = eradiate.run(exp2, spp=spp2)

    # The test should pass, even considering a large significance level
    test = tt.IndependantStudentTTest(
        name="type_I_error_test",
        value=result1,
        reference=result2,
        threshold=0.9,
        archive_dir="tests/",
        variable="radiance",
    )

    passed, p_value = test._evaluate()

    assert passed


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
@pytest.mark.slow
@pytest.mark.parametrize("spp", [64, 256, 1024])
@pytest.mark.parametrize("sigma_scale", [2.0, 3.0])
def test_t_test_biased_simulation(mode_ckd_double, spp, sigma_scale):
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

    result2["radiance"] = result2.radiance + sigma_scale * np.sqrt(
        result1.radiance_var + result2.radiance_var
    )

    # The test should not pass, even considering a small significance level
    test = tt.IndependantStudentTTest(
        name="type_II_error_test",
        value=result1,
        reference=result2,
        threshold=0.001,
        archive_dir="tests/",
        variable="radiance",
    )

    passed, p_value = test._evaluate()

    assert not passed
