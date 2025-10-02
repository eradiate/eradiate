import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.regression import RMSETest
from eradiate.test_tools.test_cases.rami4atm import (
    create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp,
)
from eradiate.test_tools.util import append_doc


@pytest.fixture
def exp():
    yield create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()


@append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp, prepend=True)
@pytest.mark.regression
def test_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp(
    mode_ckd_double, artefact_dir, session_timestamp, exp, plot_figures
):
    r"""
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version. The reference was compared against the libRadtran and RTMOM
    radiative transfer models. Comparison is done with a RMSE test with
    a threshold of 0.005.
    """

    result = eradiate.run(exp)
    logger.info(result._repr_html_(), html=True)

    test = RMSETest(
        name=(
            f"{session_timestamp:%Y%m%d-%H%M%S}-"
            "rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp"
        ),
        value=result,
        reference="tests/regression_test_references/rami4atm/"
        "hom00_bla_sd2s_m03_z30a000_brfpp-ref.nc",
        threshold=0.005,
        archive_dir=artefact_dir,
        variable="brf_srf",
        plot=plot_figures,
    )

    assert test.run()
