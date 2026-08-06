import numpy as np
import pytest

import eradiate
from eradiate.test_tools.report import report_logger
from eradiate.test_tools.test_cases import rami4atm

#: Sample count shared by every case of the suite
SPP = 1_000


@pytest.mark.regression
def test_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp(mode_ckd_double):
    r"""
    *RAMI4ATM HOM00_BLA_A00S_M04*

    This scenario is based on the ``HOM00_BLA_A00S_M04_z30a000-brfpp`` scenario
    of the RAMI4ATM benchmark.

    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile, absorption only
    - Surface: Black
    - Illumination: Directional illumination with a zenith angle of 30°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 660 nm

    *Expected behaviour*

    A black surface under a purely absorbing atmosphere reflects nothing, so
    the BRF is zero. This is an analytic check, which is why the case is not
    part of :data:`.rami4atm.CASES`: it needs no reference dataset.
    """
    exp = rami4atm.create_toa("hom00_bla_a00s_m04_z30a000_brfpp", spp=SPP)

    result = eradiate.run(exp)
    report_logger.html(result._repr_html_())
    assert np.allclose(result.brf_srf, 0.0)


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case_id", rami4atm.CASES)
@pytest.mark.filterwarnings(
    "ignore:User-specified background spectral grid is overridden by atmosphere spectral grid"
)
def test_rami4atm(mode_ckd_double, case_id, dataset_regression):
    """
    *RAMI4ATM benchmark scenarios*

    Compare the simulation results of the current branch against stored
    references, for every scenario of :data:`.rami4atm.CASES`.

    *Expected behaviour*

    Each case carries its own criteria (see :class:`.rami4atm.Case`); all of
    them must be met.
    """
    case = rami4atm.CASES[case_id]

    results = [eradiate.run(exp) for exp in case.make_experiments(spp=SPP)]
    result = case.postprocess(results)
    report_logger.html(result._repr_html_())

    # All tested variables share a single reference dataset
    dataset_regression.check(result, case.tests, basename=f"rami4atm/{case.id}-ref")
