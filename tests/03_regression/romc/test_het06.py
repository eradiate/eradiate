import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.test_cases.romc import create_het06_brfpp
from eradiate.test_tools.util import append_doc


@pytest.fixture
def exp():
    yield create_het06_brfpp()


@append_doc(create_het06_brfpp, prepend=True)
@pytest.mark.regression
def test_het06_brfpp(
    mode_mono_double, artefact_dir, session_timestamp, exp, plot_figures
):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version and validated with ROMC. Comparison is done with a chi-squared test
    with a threshold of 0.05.
    """
    result = eradiate.run(exp)

    logger.info(result._repr_html_(), html=True)

    test = ZTest(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-het06.nc",
        value=result,
        reference="tests/regression_test_references/het06_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
        variable="radiance",
        plot=plot_figures,
    )

    assert test.run()
