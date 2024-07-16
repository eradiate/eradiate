import pytest

import eradiate
from eradiate.test_tools.regression import Chi2Test
from eradiate.test_tools.test_cases.atmospheres import create_rpv_afgl1986_brfpp
from eradiate.test_tools.util import append_doc


@append_doc(create_rpv_afgl1986_brfpp)
@pytest.mark.regression
def test_rpv_afgl1986_brfpp(
    mode_ckd_double,
    artefact_dir,
    session_timestamp,
    absorption_database_error_handler_config,
):
    r"""
    RPV AFGL1986 regression test
    ====================================

    This is a regression test, which compares the simulation results of the
    current branch to an older reference version.

     Expected behaviour
    ------------------

    This test uses the Chi-squared criterion with a threshold of 0.05.

    """
    exp = create_rpv_afgl1986_brfpp(absorption_database_error_handler_config)
    result = eradiate.run(exp, spp=10000)

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-rpv_afgl1986.nc",
        value=result,
        reference="tests/regression_test_references/rpv_afgl1986_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
