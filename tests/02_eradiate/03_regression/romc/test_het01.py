import numpy as np
import pytest

import eradiate
from eradiate.data import data_store
from eradiate.experiments import CanopyExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_het01_brfpp(mode_mono_double, artefact_dir, session_timestamp):
    r"""
    Floating spheres (HET01) regression test
    ========================================

    This is a regression test, which compares the simulation results of the
    current branch to an older reference version.

    Rationale
    ---------

    This test case implements a basic canopy scene:

    * Surface with lambertian reflectance
    * No atmosphere
    * Three dimensional canopy

    Parameters

    * Surface: Square surface with labmertian BSDF with :math:`r = 0.159`
    * Canopy: Floating spheres made up of disks with bilambertian bsdf model
      Leaf reflectance is 0.4957, transmittance is 0.4409.
      Disk and sphere positioning follow the HET01 scenario of the RAMI-3 benchmark
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Distant reflectance measure, covering a plane, (76 angular points,
      10000 samples per pixel)

    Expected behaviour
    ------------------

    This test uses the Chi-squared criterion with a threshold of 0.05.

    """

    leaf_spec_path = data_store.fetch(
        "tests/regression_test_specifications/het01/het01_UNI_sphere.def"
    )
    leaf_pos_path = data_store.fetch(
        "tests/regression_test_specifications/het01/het01_UNI_instances.def"
    )

    exp = CanopyExperiment(
        canopy={
            "type": "discrete_canopy",
            "instanced_canopy_elements": [
                {
                    "construct": "from_file",
                    "filename": leaf_pos_path,
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "from_file",
                        "filename": leaf_spec_path,
                        "leaf_reflectance": 0.4957,
                        "leaf_transmittance": 0.4409,
                        "id": "spherical_leaf_cloud",
                    },
                },
            ],
            "size": [100, 100, 30] * ureg.m,
        },
        surface={"type": "lambertian", "reflectance": 0.159},
        padding=20,
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "spp": 10000,
                "azimuth": 180 * ureg.deg,
                "zeniths": np.arange(-75, 75.01, 2) * ureg.deg,
                "target": {
                    "type": "rectangle",
                    "xmin": -50 * ureg.m,
                    "xmax": 50 * ureg.m,
                    "ymin": -50 * ureg.m,
                    "ymax": 50 * ureg.m,
                    "z": 30 * ureg.m,
                },
            }
        ],
        illumination={
            "type": "directional",
            "zenith": 20 * ureg.deg,
            "azimuth": 0.0 * ureg.deg,
            "irradiance": 20.0,
        },
        integrator={"type": "path", "max_depth": -1},
    )

    result = eradiate.run(exp)

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-het01.nc",
        value=result,
        reference="tests/regression_test_references/het01_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
