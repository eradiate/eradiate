import pytest

import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.test_cases.atmospheres import create_rpv_afgl1986_brfpp
from eradiate.test_tools.util import append_doc


@pytest.fixture
def exp(absorption_database_error_handler_config):
    yield create_rpv_afgl1986_brfpp(absorption_database_error_handler_config)


@append_doc(create_rpv_afgl1986_brfpp, prepend=True)
def test_rpv_afgl1986_brfpp(mode_ckd_double, exp, dataset_regression):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version. Comparison is done with a Z-test with a threshold of 0.05.
    """
    result = eradiate.run(exp, spp=10000)

    dataset_regression.check(
        result,
        ZTest(0.05, variable="radiance"),
        basename="rpv_afgl1986_brfpp_ref",
    )
