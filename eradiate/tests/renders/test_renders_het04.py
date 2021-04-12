import numpy as np
import os

import eradiate
eradiate.set_mode("mono_double")

from eradiate.solvers.rami import RamiSolverApp
from eradiate._units import unit_registry as ureg

from eradiate.kernel.core.xml import load_dict

from util import bitmap_extract, z_test

eradiate_dir = os.environ["ERADIATE_DIR"]

def test_render_het04():
    """
    Test the computational results by rendering a pre defined scene and comparing
    the results with a reference.

    Here we employ a statistical test method, the z-test. By computing for
    each pixel the probability of agreeing with the reference values we get a
    measure of the overall agreement. If at least 95% of pixels agree with
    the reference, we accept the result and pass the test.
    """

    spp = 512000

    scene_dict = {
            "canopy": {
                "type": "discrete_canopy",
                "instanced_leaf_clouds": [
                    {
                        "construct": "from_file",
                        "filename": "/home/mopsi/src/rayference/eradiate/resources/tests/HET04/HET04_sph_positions.def",
                        "leaf_cloud": {
                            "construct": "from_file",
                            "filename": "/home/mopsi/src/rayference/eradiate/resources/tests/HET04/HET04_sph.def",
                            "leaf_reflectance": 0.49,
                            "leaf_transmittance": 0.41,
                            "id": "spherical_leaf_cloud"
                        },
                    },
                    {
                        "construct": "from_file",
                        "filename": "/home/mopsi/src/rayference/eradiate/resources/tests/HET04/HET04_cyl_positions.def",
                        "leaf_cloud": {
                            "construct": "from_file",
                            "filename": "/home/mopsi/src/rayference/eradiate/resources/tests/HET04/HET04_cyl.def",
                            "leaf_reflectance": 0.45,
                            "leaf_transmittance": 0.3,
                            "id": "cylindrical_leaf_cloud"
                        }
                    }
                ],
                "size": [270, 270, 30] * ureg.m
            },
            "surface": {
                "type": "lambertian",
                "reflectance": 0.15
            },
            "padding": 20,
            "measures": [{
                "type": "distant",
                "spp": spp,
                "target": {
                    "type": "rectangle",
                    "xmin": -135.0*ureg.m,
                    "xmax":135.0*ureg.m,
                    "ymin": -135.0*ureg.m,
                    "ymax": 135.0*ureg.m
                },
                "film_resolution": (45, 1),
                "orientation": (0 + 180)*ureg.deg
            }],
            "illumination": {
                "type": "directional",
                "zenith": 20*ureg.deg,
                "irradiance": 20.
            },
            "integrator": {
                "type": "path",
                "max_depth": -1
            }
        }

    app_inf_pplane = RamiSolverApp(scene_dict)
    scene_dict = app_inf_pplane.scene.kernel_dict()
    inner_integrator = scene_dict.pop("integrator")
    scene_dict["integrator"] = {
        "type": "moment",
        "subintegrator": inner_integrator
    }
    scene = load_dict(dict(scene_dict))
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)

    bmp = scene.sensors()[0].film().bitmap(raw=False)
    img, var_img = bitmap_extract(bmp)


    ref_data = np.load(f"{eradiate_dir}/resources/data/tests/test_renders/HET04/HET04.npz")
    ref_img = ref_data["img"]
    ref_var_img = ref_data["var_img"]


    significance_level = 0.01
    p_value = z_test(img, spp, ref_img, ref_var_img)

    # this correction accounts for the fact that the probability for a test to fail by chance
    # increases with the number of pixels
    alpha = 1.0 - (1.0 - significance_level) ** (1.0 / 45.)
    success = (p_value > alpha)

    # 2 out of 45 is roughly 5% of tests
    # we fail the test is more than 5% of pixels deviate from the reference
    if sum(int(i) for i in success) < 43:
        print(f"{45-sum(int(i) for i in success)} pixels deviated. Test failed.")
        import matplotlib.pyplot as plt
        output_dir = f"{eradiate_dir}/eradiate/tests/renders/output"
        if not os.path.exists(f"{eradiate_dir}/eradiate/tests/renders/output"):
            os.mkdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, "HET04.png")):
            os.remove(os.path.join(output_dir, "HET04.png"))
        fig, ax = plt.subplots(2, 2, figsize=(5,5), dpi=120)
        # plt.tight_layout()
        x = np.arange(-89, 90, 4)

        ax[0][0].plot(x, img)
        ax[0][0].set_title("Image")
        ax[0][0].set_ylabel("Leaving radiance")
        ax[0][0].set_xlabel("VZA")
        ax[0][1].plot(x, ref_img)
        ax[0][1].set_title("Reference image")
        ax[0][1].set_xlabel("VZA")
        ax[1][0].plot(x, var_img)
        ax[1][0].set_title("Variance")
        ax[1][0].set_xlabel("VZA")
        ax[1][0].set_ylabel("Sample variance")
        ax[1][1].plot(x, ref_var_img)
        ax[1][1].set_title("Reference variance")
        ax[1][1].set_xlabel("VZA")

        plt.savefig(f"{eradiate_dir}/eradiate/tests/renders/output/HET04.png")
        assert False
