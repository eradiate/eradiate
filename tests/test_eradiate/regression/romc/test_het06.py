import os

import numpy as np
import pytest
import xarray as xr

from eradiate.data import data_store
from eradiate.experiments import RamiExperiment
from eradiate.test_tools.regression import Chi2Test
from eradiate.units import unit_registry as ureg


@pytest.mark.regression
def test_het06_brfpp(mode_mono_double, metadata, session_timestamp):
    """
    Coniferous forest no topography (HET06)

    This test case uses the coniferous forest scene with no topography
    from RAMI-3.
    It uses the definition files as provided on the RAMI-3 website for
    placement of leaves and leaf clouds.

    The remaining parameters are:

    - Leaf reflectance: 0.08
    - Leaf transmittance: 0.03
    - Soil reflectance: 0.86
    - Sun zenith angle: 20°
    - Sun azimuth angle: 0°

    This test uses the Chi-squared criterion with a threshold of 0.05
    """

    tree_pos_path = data_store.fetch(
        "tests/regression_test_specifications/het06/het06_scene.def"
    )

    exp = RamiExperiment(
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
                "construct": "from_viewing_angles",
                "spp": 10000,
                "azimuths": [180] * ureg.deg,
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

    exp.run()
    result = exp.results["measure"]

    reference_path = data_store.fetch(
        "tests/regression_test_references/het06_brfpp_ref.nc"
    )
    reference = xr.load_dataset(reference_path)
    archive_path = metadata.get("archive_path", None)

    archive_filename = (
        os.path.join(archive_path, f"{session_timestamp:%Y%m%d-%H%M%S}-het06.nc")
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