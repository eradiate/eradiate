import pytest

import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.report import report_logger
from eradiate.test_tools.test_cases.romc import create_het06_brfpp
from eradiate.test_tools.util import append_doc


@pytest.fixture
def exp():
    yield create_het06_brfpp()


@append_doc(create_het06_brfpp, prepend=True)
@pytest.mark.regression
def test_het06_brfpp(mode_mono_double, exp, dataset_regression):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version and validated with ROMC. Comparison is done with a Z-test with a
    threshold of 0.05.
    """
    result = eradiate.run(exp)

    report_logger.html(result._repr_html_())

    dataset_regression.check(
        result,
        ZTest(0.05, variable="radiance"),
        basename="het06_brfpp_ref",
    )
