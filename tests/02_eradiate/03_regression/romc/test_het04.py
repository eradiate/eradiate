import os

import numpy as np
import pytest
import xarray as xr

from eradiate.contexts import KernelDictContext
from eradiate.data import data_store
from eradiate.experiments import RamiExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_het04a1_brfpp(mode_mono_double, metadata, session_timestamp):
    """
    Real zoom in (HET04a1)

    This test case uses the real zoom in scene from RAMI-3.
    It uses the definition files as provided on the RAMI-3 website for
    placement of leaves and leaf clouds.

    The remaining parameters are:

    - Leaf reflectance spheres: 0.49
    - Leaf transmittance spheres: 0.41
    - Leaf reflectance cylinders: 0.45
    - Leaf transmittance cylinders: 0.3
    - Soil reflectance: 0.15
    - Sun zenith angle: 20°
    - Sun azimuth angle: 0°

    This test uses the Chi-squared criterion with a threshold of 0.05
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

    exp = RamiExperiment(
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
                "construct": "from_viewing_angles",
                "spp": 10000,
                "azimuths": [180] * ureg.deg,
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

    kd = exp.kernel_dict(ctx=KernelDictContext())
    tran = kd.data["spherical_leaf_cloud"]["spherical_leaf_cloud_leaf_1"]["to_world"]

    exp.run()
    result = exp.results["measure"]

    reference_path = data_store.fetch(
        "tests/regression_test_references/het04_brfpp_ref.nc"
    )
    reference = xr.load_dataset(reference_path)
    archive_path = metadata.get("archive_path", None)

    archive_filename = (
        os.path.join(archive_path, f"{session_timestamp:%Y%m%d-%H%M%S}-het04.nc")
        if archive_path
        else None
    )

    test = Chi2Test(
        value=result,
        reference=reference,
        threshold=0.05,
        archive_filename=archive_filename,
    )

    assert test.run()
