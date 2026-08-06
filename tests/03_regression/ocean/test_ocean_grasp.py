import pytest

import eradiate
from eradiate.test_tools.regression import RMSETest, ZTest
from eradiate.test_tools.test_cases.ocean import (
    create_ocean_grasp_coastal_no_atm,
    create_ocean_grasp_open_atm,
    create_ocean_grasp_open_no_atm,
)


@pytest.mark.regression
def test_ocean_grasp_coastal_no_atm(mode_mono_double, dataset_regression):
    """
    *Ocean GRASP Coastal no atm regression test*

    Compare the simulation results of the current branch to results directly
    validated against the GRASP model. This test targets a coastal ocean
    scenario and does not include an atmosphere.

    *Expected behaviour*

    This test uses the RMSE criterion with a threshold of 10⁻⁶, applied
    independently to each wavelength.
    """
    exp = create_ocean_grasp_coastal_no_atm()
    result = eradiate.run(exp)

    dataset_regression.check(
        result,
        RMSETest(1e-6, variable="brf", dim="w"),
        basename="ocean_grasp_REF_OC_NN00_I_S20_PPL",
    )


@pytest.mark.regression
def test_ocean_grasp_open_no_atm(mode_mono_double, dataset_regression):
    """
    *Ocean GRASP Open no atm regression test*

    Compare the simulation results of the current branch to results directly
    validated against the GRASP model. This test targets an open ocean scenario
    and does not include an atmosphere.

    *Expected behaviour*

    This test uses the RMSE criterion with a threshold of 10⁻⁶, applied
    independently to each wavelength.
    """
    exp = create_ocean_grasp_open_no_atm()
    result = eradiate.run(exp)

    dataset_regression.check(
        result,
        RMSETest(1e-6, variable="brf", dim="w"),
        basename="ocean_grasp_REF_OO_NN00_I_S20_PPL",
    )


@pytest.mark.regression
def test_ocean_grasp_open_atm(mode_mono_double, dataset_regression):
    """
    *Ocean GRASP Open atm regression test*

    Compares the simulation results of the current branch to results directly
    validated against the GRASP model. This test targets a coastal ocean
    scenario and includes an atmosphere.

    *Expected behaviour*

    This test uses the z-test criterion with a threshold of 0.01.
    """
    exp = create_ocean_grasp_open_atm()
    result = eradiate.run(exp, spp=int(1e5))

    dataset_regression.check(
        result,
        ZTest(0.01, variable="radiance"),
        basename="ocean_grasp_REF_OO_UB01_I_S20_PPL",
    )
