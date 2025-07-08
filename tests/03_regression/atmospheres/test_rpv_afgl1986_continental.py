import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.test_cases.atmospheres import (
    create_rpv_afgl1986_continental_brfpp,
)
from eradiate.test_tools.util import append_doc


@pytest.fixture
def exp(absorption_database_error_handler_config):
    yield create_rpv_afgl1986_continental_brfpp(
        absorption_database_error_handler_config
    )


@append_doc(create_rpv_afgl1986_continental_brfpp, prepend=True)
def test_rpv_afgl1986_continental_brfpp(
    mode_ckd_double, artefact_dir, session_timestamp, exp
):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version. Comparison is done with a chi-squared test with a threshold of
    0.05.
    """
    result = eradiate.run(exp, spp=10000)
    logger.info(result._repr_html_(), html=True)

    test = ZTest(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-rpv_afgl1986_continental.nc",
        value=result,
        reference="tests/regression_test_references/rpv_afgl1986_continental_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
        variable="radiance",
    )

    assert test.run()
