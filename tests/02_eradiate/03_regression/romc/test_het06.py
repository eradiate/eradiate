import numpy as np
import pytest

import eradiate
from eradiate.data import data_store
from eradiate.experiments import CanopyExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_het06_brfpp(mode_mono_double, artefact_dir, session_timestamp):
    r"""
    Coniferous forest (HET06) regression test
    =========================================

    This is a regression test, which compares the simulation results of the
    current branch to an older reference version.

    Rationale
    ---------

    This test case implements a basic canopy scene:

    * Surface with lambertian reflectance
    * No atmosphere
    * Three dimensional canopy

    Parameters

    * Surface: Square surface with labmertian BSDF with :math:`r = 0.86`
    * Canopy: Coniferous trees made up of a conical leaf cloud and a cylindrical
      trunk.

      * Leaf reflectance: 0.08
      * Leaf transmittance: 0.03
      * Trunk reflectance: 0.14

      Disk and tree positioning follow the HET06 scenario of the RAMI-3 benchmark
    * Illumination: Directional illumination with a zenith angle :math:`\theta = 20°`
    * Sensor: Distant reflectance measure, covering a plane, (76 angular points,
      10000 samples per pixel)

    Expected behaviour
    ------------------

    This test uses the Chi-squared criterion with a threshold of 0.05.

    """

    tree_pos_path = data_store.fetch(
        "tests/regression_test_specifications/het06/het06_scene.def"
    )

    exp = CanopyExperiment(
        canopy={
            "type": "discrete_canopy",
            "instanced_canopy_elements": [
                {
                    "construct": "from_file",
                    "filename": tree_pos_path,
                    "canopy_element": {
                        "type": "abstract_tree",
                        "trunk_height": 1.5 * ureg.m,
                        "trunk_radius": 0.15 * ureg.m,
                        "trunk_reflectance": 0.14,
                        "leaf_cloud_extra_offset": [0.0, 0.0, 0.0] * ureg.m,
                        "id": "conifer_tree",
                        "leaf_cloud": {
                            "construct": "cone",
                            "l_vertical": 12.0 * ureg.m,
                            "leaf_radius": 0.05,
                            "radius": 1.8 * ureg.m,
                            "n_leaves": 6480,
                            "leaf_reflectance": 0.08,
                            "leaf_transmittance": 0.03,
                            "id": "conical_leaf_cloud",
                        },
                    },
                }
            ],
            "size": [500, 500, 13.5] * ureg.m,
        },
        surface={"type": "lambertian", "reflectance": 0.86},
        padding=0,
        measures=[
            {
                "type": "mdistant",
                "construct": "hplane",
                "spp": 10000,
                "azimuth": 180 * ureg.deg,
                "zeniths": np.arange(-75, 75.01, 2) * ureg.deg,
                "target": {
                    "type": "rectangle",
                    "xmin": -250 * ureg.m,
                    "xmax": 250 * ureg.m,
                    "ymin": -250 * ureg.m,
                    "ymax": 250 * ureg.m,
                    "z": 6.75 * ureg.m,
                },
            }
        ],
        illumination={
            "type": "directional",
            "zenith": 40 * ureg.deg,
            "azimuth": 0.0 * ureg.deg,
            "irradiance": 20.0,
        },
        integrator={"type": "path", "max_depth": -1},
    )

    result = eradiate.run(exp)

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-het06.nc",
        value=result,
        reference="tests/regression_test_references/het06_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
