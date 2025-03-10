import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.regression import Chi2Test
from eradiate.test_tools.test_cases.rami4atm import (
    create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp,
)
from eradiate.test_tools.util import append_doc


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
@pytest.mark.regression
def test_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp(
    mode_ckd_double, artefact_dir, session_timestamp
):
    r"""
    RAMI4ATM HOM00_BLA_SD2S_M03 regression test
    ===========================================

    This regression test uses the ``HOM00_BLA_SD2S_M03_z30a000-brfpp`` scenario
    of the RAMI4ATM benchmark. The reference solution is trusted and compared
    against the libRadtran and RTMOM radiative transfer models.

    Expected behaviour
    ------------------
    This test uses the Chi-squared criterion with a threshold of 0.05.

    """

    exp = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
    result = eradiate.run(exp)

    logger.info(result._repr_html_(), html=True)

    test = Chi2Test(
        name=(
            f"{session_timestamp:%Y%m%d-%H%M%S}-"
            "rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp"
        ),
        value=result,
        reference=(
            "tests/regression_test_references/"
            "rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp_ref.nc"
        ),
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
