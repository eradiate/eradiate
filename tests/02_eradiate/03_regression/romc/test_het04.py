import numpy as np
import pytest

import eradiate
from eradiate.data import data_store
from eradiate.experiments import CanopyExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


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

    leaf_spec_path_sph = data_store.fetch(
        "tests/regression_test_specifications/het04/het04_sph.def"
    )
    leaf_pos_path_sph = data_store.fetch(
        "tests/regression_test_specifications/het04/het04_sph_positions.def"
    )
    leaf_spec_path_cyl = data_store.fetch(
        "tests/regression_test_specifications/het04/het04_cyl.def"
    )
    leaf_pos_path_cyl = data_store.fetch(
        "tests/regression_test_specifications/het04/het04_cyl_positions.def"
    )

    exp = CanopyExperiment(
        canopy={
            "type": "discrete_canopy",
            "instanced_canopy_elements": [
                {
                    "construct": "from_file",
                    "filename": leaf_pos_path_sph,
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "from_file",
                        "filename": leaf_spec_path_sph,
                        "leaf_reflectance": 0.49,
                        "leaf_transmittance": 0.41,
                        "id": "spherical_leaf_cloud",
                    },
                },
                {
                    "construct": "from_file",
                    "filename": leaf_pos_path_cyl,
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "from_file",
                        "filename": leaf_spec_path_cyl,
                        "leaf_reflectance": 0.45,
                        "leaf_transmittance": 0.3,
                        "id": "cylindrical_leaf_cloud",
                    },
                },
            ],
            "size": [270, 270, 15] * ureg.m,
        },
        surface={"type": "lambertian", "reflectance": 0.15},
        padding=5,
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "spp": 20000,
                "azimuth": 180 * ureg.deg,
                "zeniths": np.arange(-75, 75.01, 2) * ureg.deg,
                "target": {
                    "type": "rectangle",
                    "xmin": -135 * ureg.m,
                    "xmax": 135 * ureg.m,
                    "ymin": -135 * ureg.m,
                    "ymax": 135 * ureg.m,
                    "z": 15 * ureg.m,
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
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-het04.nc",
        value=result,
        reference="tests/regression_test_references/het04_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
