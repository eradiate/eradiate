import numpy as np

from eradiate import fresolver
from eradiate.experiments import CanopyExperiment
from eradiate.units import unit_registry as ureg


def fetch_het01_brfpp():
    """
    Fetches data necessary for test case het01_brfpp.
    Note that this function can be used to prefetch data
    before running a test case.

    Returns
    -------
    dict
        dictionary containing the various data paths.
    """

    data = {}
    data["leaf_spec_path"] = fresolver.resolve(
        "tests/regression_test_specifications/het01/het01_UNI_sphere.def"
    )
    data["leaf_pos_path"] = fresolver.resolve(
        "tests/regression_test_specifications/het01/het01_UNI_instances.def"
    )

    return data


def create_het01_brfpp(prefetched_data=None):
    """
    *ROMC HET01 (floating spheres)*

    *Scene setup*

    - Surface: Lambertian BSDF with r = 0.159.
    - Canopy:
      (a) Floating spheres made up of disks with bilambertian BSDF model;
      leaf reflectance 0.4957, transmittance 0.4409.
      (b) Disk and sphere positions follow the HET01 scenario of the RAMI-3
      benchmark.
    - Illumination: Directional illumination with a zenith angle of 20°.
    - Sensor: Multi-distant measure covering the principal plane,
      from -75° to 75° with 2° increments, delta SRF positioned at λ = 550 nm.
    """
    if prefetched_data is None:
        prefetched_data = fetch_het01_brfpp()

    leaf_spec_path = prefetched_data["leaf_spec_path"]
    leaf_pos_path = prefetched_data["leaf_pos_path"]

    return CanopyExperiment(
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
        integrator={"type": "path", "max_depth": -1, "moment": True},
    )


def fetch_het04a1_brfpp():
    """
    Fetches data necessary for test case het04_brfpp.
    Note that this function can be used to prefetch data
    before running a test case.

    Returns
    -------
    dict
        dictionary containing the various data paths.
    """

    data = {}

    data["leaf_spec_path_sph"] = fresolver.resolve(
        "tests/regression_test_specifications/het04/het04_sph.def"
    )
    data["leaf_pos_path_sph"] = fresolver.resolve(
        "tests/regression_test_specifications/het04/het04_sph_positions.def"
    )
    data["leaf_spec_path_cyl"] = fresolver.resolve(
        "tests/regression_test_specifications/het04/het04_cyl.def"
    )
    data["leaf_pos_path_cyl"] = fresolver.resolve(
        "tests/regression_test_specifications/het04/het04_cyl_positions.def"
    )

    return data


def create_het04a1_brfpp(prefetched_data=None):
    """
    *ROMC HET04a1 (Real zoom in)*

    This test implements variant a1 of the HET04 scenario, in which the entire
    scene is targeted by the sensor.

    *Scene setup*

    - Surface: Lambertian BSDF with r = 0.15
    - Canopy:
      (a) Floating spheres made up of disks with bilambertian bsdf model;
      leaf reflectance 0.49, transmittance 0.41.
      (b) Floating cylinders made up of disks with bilambertian bsdf model;
      leaf reflectance 0.45, transmittance 0.43.
      (c) Disk, sphere and cylinder positioning follow the HET04 scenario of the
      RAMI-3 benchmark.
    - Illumination: Directional illumination with a zenith angle of 20°.
    - Sensor: Multi-distant measure covering the principal plane, from -75° to
      75° with 2° increments, delta SRF positioned at λ = 550 nm.
    """
    if prefetched_data is None:
        prefetched_data = fetch_het04a1_brfpp()

    leaf_spec_path_sph = prefetched_data["leaf_spec_path_sph"]
    leaf_pos_path_sph = prefetched_data["leaf_pos_path_sph"]
    leaf_spec_path_cyl = prefetched_data["leaf_spec_path_cyl"]
    leaf_pos_path_cyl = prefetched_data["leaf_pos_path_cyl"]

    return CanopyExperiment(
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
        integrator={"type": "path", "max_depth": -1, "moment": True},
    )


def fetch_het06_brfpp():
    """
    Fetches data necessary for test case het04_brfpp.
    Note that this function can be used to prefetch data
    before running a test case.

    Returns
    -------
    dict
        dictionary containing the various data paths.
    """

    data = {}

    data["tree_pos_path"] = fresolver.resolve(
        "tests/regression_test_specifications/het06/het06_scene.def"
    )

    return data


def create_het06_brfpp(prefetched_data=None):
    """
    *ROMC HET06 (Coniferous forest)*

    *Scene setup*

    - Surface: Lambertian BSDF with r = 0.86.
    - Canopy: Coniferous trees made up of a conical leaf cloud and a cylindrical
      trunk (leaf reflectance: 0.08; leaf transmittance: 0.03; trunk
      reflectance: 0.14). Disk and tree positioning follow the HET06 scenario of
      the RAMI-3 benchmark.
    - Illumination: Directional illumination with a zenith angle of 20°.
    - Sensor: Multi-distant measure covering the principal plane, from -75° to
      75° with 2° increments, delta SRF positioned at λ = 550 nm.
    """

    if prefetched_data is None:
        prefetched_data = fetch_het06_brfpp()

    tree_pos_path = prefetched_data["tree_pos_path"]

    return CanopyExperiment(
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
        integrator={"type": "path", "max_depth": -1, "moment": True},
    )
