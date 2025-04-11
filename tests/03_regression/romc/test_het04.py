import pytest
from robot.api import logger

import eradiate
from eradiate.test_tools.regression import Chi2Test
from eradiate.test_tools.test_cases.romc import create_het04a1_brfpp
from eradiate.test_tools.util import append_doc


@append_doc(create_het04a1_brfpp)
@pytest.mark.regression
def test_het04a1_brfpp(mode_mono_double, artefact_dir, session_timestamp):
    r"""
    Real zoom in (HET04a1) regression test
    ======================================

    This is a regression test, which compares the simulation results of the
    current branch to an older reference version.

    Rationale
    ---------

    This test case implements a basic canopy scene:

    * Surface with lambertian reflectance
    * No atmosphere
    * Three dimensional canopy

    Parameters

    * Surface: Square surface with labmertian BSDF with :math:`r = 0.15`
    * Canopy:

      * Floating spheres made up of disks with bilambertian bsdf model.
        Leaf reflectance is 0.49, transmittance is 0.41.
      * Floating cylinders made up of disks with bilambertian bsdf model.
        Leaf reflectance is 0.45, transmittance is 0.43.

      Disk, sphere and cylinder positioning follow the HET04 scenario of the
      RAMI-3 benchmark.
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Distant reflectance measure, covering a plane, (76 angular points,
      10000 samples per pixel). This test implements the variant a1 of the HET04
      scenario, in which the entire scene is targeted by the sensor.

    Expected behaviour
    ------------------

    This test uses the Chi-squared criterion with a threshold of 0.05.

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
    )

    assert test.run()
