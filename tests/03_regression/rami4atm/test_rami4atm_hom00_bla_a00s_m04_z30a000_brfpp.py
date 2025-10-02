import numpy as np
import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.test_cases.rami4atm import (
    create_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp,
)
from eradiate.test_tools.util import append_doc


@append_doc(create_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp, prepend=True)
@pytest.mark.regression
def test_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp(mode_ckd_double):
    r"""
    *Expected behaviour*

    The atmosphere only absorbs and the surface is black.
    Simulation results should be close to zero.
    """

    exp = create_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp()
    result = eradiate.run(exp)
    logger.info(result._repr_html_(), html=True)

    assert np.allclose(result.brf_srf, 0.0)
