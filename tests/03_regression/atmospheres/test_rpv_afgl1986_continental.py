import pytest

import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.report import report_logger
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
    mode_ckd_double,
    exp,
    dataset_regression,
):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version. Comparison is done with a Z-test with a threshold of 0.05.
    """
    result = eradiate.run(exp, spp=10000)
    report_logger.html(result._repr_html_())

    dataset_regression.check(
        result,
        ZTest(0.05, variable="radiance"),
        basename="rpv_afgl1986_continental_brfpp_ref",
    )
