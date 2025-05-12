import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.test_cases.atmospheres import create_rpv_afgl1986_brfpp
from eradiate.test_tools.util import append_doc


@append_doc(create_rpv_afgl1986_brfpp, prepend=True)
def test_rpv_afgl1986_brfpp(
    mode_ckd_double,
    artefact_dir,
    session_timestamp,
    absorption_database_error_handler_config,
):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with a prior
    version. Comparison is done with a chi-squared test with a threshold of
    0.05.
    """
    exp = create_rpv_afgl1986_brfpp(absorption_database_error_handler_config)
    result = eradiate.run(exp, spp=10000)

    test = ZTest(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-rpv_afgl1986.nc",
        value=result,
        variable="radiance",
        reference="tests/regression_test_references/rpv_afgl1986_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
