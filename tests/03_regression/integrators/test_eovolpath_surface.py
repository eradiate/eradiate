import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.report import report_logger
from eradiate.test_tools.test_cases.integrators import (
    create_eovolpath_surface,
    create_volpath_surface,
)
from eradiate.test_tools.util import append_doc


@append_doc(create_eovolpath_surface, prepend=True)
def test_eovolpath_surface(
    mode_ckd_double,
    absorption_database_error_handler_config,
    dataset_regression,
):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with the standard
    ``volpath`` integrator. Comparison is done with a Z-test with a threshold
    of 0.05.
    """

    if dataset_regression.reference_path("eovolpath_surface_ref") is None:
        # Ensure that volpath is used when creating a reference
        exp = create_volpath_surface(absorption_database_error_handler_config)
        spp = int(1e5)
    else:
        exp = create_eovolpath_surface(absorption_database_error_handler_config)
        spp = int(1e4)

    result = eradiate.run(exp, spp=spp)
    report_logger.html(result._repr_html_())

    dataset_regression.check(
        result,
        ZTest(0.05, variable="radiance"),
        basename="eovolpath_surface_ref",
    )
