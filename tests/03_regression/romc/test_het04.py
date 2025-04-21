import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.regression import Chi2Test
from eradiate.test_tools.test_cases.romc import create_het04a1_brfpp
from eradiate.test_tools.util import append_doc


@append_doc(create_het04a1_brfpp, prepend=True)
@pytest.mark.regression
def test_het04a1_brfpp(mode_mono_double, artefact_dir, session_timestamp):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version and validated with ROMC. Comparison is done with a chi-squared test
    with a threshold of 0.05.
    """
    exp = create_het04a1_brfpp()
    result = eradiate.run(exp)

    logger.info(result._repr_html_(), html=True)

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-het04.nc",
        value=result,
        reference="tests/regression_test_references/het04_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
        variable="brf",
    )

    assert test.run()
