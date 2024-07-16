import pytest

import eradiate
from eradiate.test_tools.regression import Chi2Test
from eradiate.test_tools.test_cases.romc import create_het06_brfpp
from eradiate.test_tools.util import append_doc


@append_doc(create_het06_brfpp)
@pytest.mark.regression
def test_het06_brfpp(mode_mono_double, artefact_dir, session_timestamp):
    r"""
    Coniferous forest (HET06) regression test
    =========================================

    This is a regression test, which compares the simulation results of the
    current branch to an older reference version.

    Expected behaviour
    ------------------

    This test uses the Chi-squared criterion with a threshold of 0.05.

    """

    exp = create_het06_brfpp()
    result = eradiate.run(exp)

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-het06.nc",
        value=result,
        reference="tests/regression_test_references/het06_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
