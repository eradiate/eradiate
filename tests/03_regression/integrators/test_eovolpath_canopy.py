from robot.api import logger

import eradiate
from eradiate.test_tools.regression import ZTest, reference_converter
from eradiate.test_tools.test_cases.integrators import (
    create_eovolpath_canopy,
    create_volpath_canopy,
)
from eradiate.test_tools.util import append_doc


@append_doc(create_eovolpath_canopy, prepend=True)
def test_eovolpath_canopy(
    mode_ckd_double,
    absorption_database_error_handler_config,
    artefact_dir,
    session_timestamp,
    plot_figures,
):
    """
    *Expected behaviour*

    Simulation results are compared to a reference obtained with the standard
    ``volpath`` integrator. Comparison is done with a Z-test with a threshold
    of 0.05.
    """

    reference = reference_converter(
        "tests/regression_test_references/eovolpath_canopy_ref.nc"
    )

    if not reference:
        # Ensure that volpath is used when creating a reference
        exp = create_volpath_canopy(absorption_database_error_handler_config)
        spp = int(1e5)
    else:
        exp = create_eovolpath_canopy(absorption_database_error_handler_config)
        spp = int(1e4)

    result = eradiate.run(exp, spp=spp)
    logger.info(result._repr_html_(), html=True)

    test = ZTest(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-eovolpath_canopy.nc",
        value=result,
        reference=reference,
        threshold=0.05,
        archive_dir=artefact_dir,
        variable="radiance",
        plot=plot_figures,
    )

    assert test.run()
