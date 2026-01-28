import numpy as np
import pytest
from robot.api import logger

import eradiate
from eradiate import fresolver
from eradiate.test_tools.regression import SidakTTest
from eradiate.test_tools.test_cases import rami4atm

cases = [c for c in rami4atm.registry if c != "hom00_bla_a00s_m04_z30a000_brfpp"]


@pytest.mark.regression
def test_rami4atm_hom00_bla_a00s_m04_z30a000_brfpp(mode_ckd_double, artefact_dir):
    r"""
    *RAMI4ATM HOM00_BLA_S00S_M04*

    This scenario is based on the ``HOM00_BLA_S00S_M04_z30a000-brfpp`` scenario
    of the RAMI4ATM benchmark.

    *Scene setup*

    - Geometry: 1D plane-parallel
    - Atmosphere: Molecular atmosphere using the AFGL 1986 (U.S. Standard) profile, absorption only
    - Surface: Black
    - Illumination: Directional illumination with a zenith angle of 30°
    - Sensor: Multi-distant measure covering the principal plane, from -75° to 75° with 2° increments, delta SRF positioned at λ = 660 nm
    """
    # TODO: This test case definition is kept for compatibility
    _, (exp,) = rami4atm.create_rami4atm_toa("hom00_bla_a00s_m04_z30a000_brfpp", 1000)

    result = eradiate.run(exp)
    logger.info(result._repr_html_(), html=True)
    assert np.allclose(result.brf_srf, 0.0)


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", cases)
@pytest.mark.filterwarnings(
    "ignore:User-specified a background spectral grid is overridden by atmosphere spectral grid"
)
def test_rami4atm(mode_ckd_double, case, artefact_dir):
    specification = rami4atm.registry[case]
    ctor = specification.get("constructor")
    postprocess = specification.get("postprocess", lambda ls, _: ls[0])
    variables = specification.get("variables", ["radiance"])
    test_ctor = specification.get("test", SidakTTest)
    threshold = specification.get("threshold")

    srf_id, exps = ctor(spp=1000)
    srf = fresolver.load_dataset(f"srf/{srf_id}.nc")

    raw_results = [eradiate.run(exp) for exp in exps]

    result = postprocess(raw_results, srf)
    logger.info(result._repr_html_(), html=True)

    reference = fresolver.load_dataset(
        f"tests/regression_test_references/rami4atm/{case}-ref.nc"
    )
    logger.info(reference._repr_html_(), html=True)

    for variable in variables:
        logger.info(f"Testing {variable}")

        test = test_ctor(
            name=case,
            value=result,
            reference=reference,
            threshold=threshold,
            archive_dir=artefact_dir,
            variable=variable,
            plot=False,
        )

        passed = test.run(diagnostic=True)

        assert passed
