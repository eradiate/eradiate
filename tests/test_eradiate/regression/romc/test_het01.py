import os

import numpy as np
import pytest
import xarray as xr

from eradiate.data import data_store
from eradiate.experiments import RamiExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_het01_brfpp(mode_mono_double, metadata, session_timestamp):
    """
    Floating disks (HET01)

    This test case uses the floating disks scene from RAMI-3.
    It uses the definition files as provided on the RAMI-3 website for
    placement of leaves and leaf clouds.

    The remaining parameters are:

    - Leaf reflectance: 0.4957
    - Leaf transmittance: 0.4409
    - Soil reflectance: 0.159
    - Sun zenith angle: 20°
    - Sun azimuth angle: 0°

    This test uses the Chi-squared criterion with a threshold of 0.05
    """

    leaf_spec_path = data_store.fetch(
        "tests/regression_test_specifications/het01/het01_UNI_sphere.def"
    )
    leaf_pos_path = data_store.fetch(
        "tests/regression_test_specifications/het01/het01_UNI_instances.def"
    )

    exp = RamiExperiment(
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
                "construct": "from_viewing_angles",
                "spp": 10000,
                "azimuths": [180] * ureg.deg,
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

    exp.run()
    result = exp.results["measure"]

    reference_path = data_store.fetch(
        "tests/regression_test_references/het01_brfpp_ref.nc"
    )
    reference = xr.load_dataset(reference_path)
    archive_path = metadata.get("archive_path", None)

    archive_filename = (
        os.path.join(archive_path, f"{session_timestamp:%Y%m%d-%H%M%S}-het01.nc")
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
