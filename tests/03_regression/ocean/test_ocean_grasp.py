import pytest

import eradiate
import eradiate.data
from eradiate.test_tools.regression import Chi2Test
from eradiate.test_tools.test_cases.ocean import (
    create_ocean_grasp_coastal_no_atm,
    create_ocean_grasp_open_atm,
    create_ocean_grasp_open_no_atm,
    ocean_grasp_wavelength,
)


@pytest.mark.regression
def test_ocean_grasp_coastal_no_atm(mode_mono_double, artefact_dir, session_timestamp):
    """
    *Ocean GRASP Coastal no atm regression test*

    Compare the simulation results of the current branch to results directly
    validated against the GRASP model. This test targets a coastal ocean
    scenario and does not include an atmosphere.

    *Expected behaviour*

    This test uses the Chi-squared criterion with a threshold of 0.05.
    """
    ref = eradiate.data.load_dataset(
        "tests/regression_test_references/ocean_grasp_REF_OC_NN00_I_S20_PPL.nc"
    )
    exp = create_ocean_grasp_coastal_no_atm()
    result = eradiate.run(exp)

    wavelength = ocean_grasp_wavelength()
    for w in wavelength:
        test = Chi2Test(
            name=(
                f"{session_timestamp:%Y%m%d-%H%M%S}-ocean_grasp_REF_OC_NN00_I_S20_PPL"
            ),
            value=result.sel(w=w),
            reference=ref.sel(w=w),
            threshold=0.05,
            archive_dir=artefact_dir,
            variable="brf",
        )

        assert test.run()


@pytest.mark.regression
def test_ocean_grasp_open_no_atm(mode_mono_double, artefact_dir, session_timestamp):
    """
    *Ocean GRASP Open no atm regression test*

    Compare the simulation results of the current branch to results directly
    validated against the GRASP model. This test targets an open ocean scenario
    and does not include an atmosphere.

    *Expected behaviour*

    This test uses the Chi-squared criterion with a threshold of 0.05.
    """
    ref = eradiate.data.load_dataset(
        "tests/regression_test_references/ocean_grasp_REF_OO_NN00_I_S20_PPL.nc"
    )
    exp = create_ocean_grasp_open_no_atm()
    result = eradiate.run(exp)

    wavelength = ocean_grasp_wavelength()
    for w in wavelength:
        test = Chi2Test(
            name=f"{session_timestamp:%Y%m%d-%H%M%S}-ocean_grasp_REF_OO_NN00_I_S20_PPL",
            value=result.sel(w=w),
            reference=ref.sel(w=w),
            threshold=0.05,
            archive_dir=artefact_dir,
            variable="brf",
        )

        assert test.run()


@pytest.mark.regression
def test_ocean_grasp_open_atm(mode_mono_double, artefact_dir, session_timestamp):
    """
    *Ocean GRASP Open atm regression test*

    Compares the simulation results of the current branch to results directly
    validated against the GRASP model. This test targets a coastal ocean
    scenario and does not include an atmosphere.

    *Expected behaviour*

    This test uses the Chi-squared criterion with a threshold of 0.05.
    """
    ref = eradiate.data.load_dataset(
        "tests/regression_test_references/ocean_grasp_REF_OO_UB01_I_S20_PPL.nc"
    )
    exp = create_ocean_grasp_open_atm()
    result = eradiate.run(exp)

    wavelength = ocean_grasp_wavelength()
    for w in wavelength:
        test = Chi2Test(
            name=f"{session_timestamp:%Y%m%d-%H%M%S}-ocean_grasp_REF_OO_UB01_I_S20_PPL",
            value=result.sel(w=w),
            reference=ref.sel(w=w),
            threshold=0.05,
            archive_dir=artefact_dir,
            variable="brf",
        )

        assert test.run()
