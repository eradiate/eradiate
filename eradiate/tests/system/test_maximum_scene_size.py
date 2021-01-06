import numpy as np
import pytest

from eradiate.util.frame import angles_to_direction


@pytest.mark.slow
def test_maximum_scene_size(mode_mono_double, json_metadata):
    r"""
    Maximum scene size (``path``)
    =============================

    This test case asserts that the maximum scene size that can be rendered without
    breaking the limits of numerical precision is above a pre set limit.


    Rationale
    ---------

        - Geometry: a square surface with size in a series of powers of ten (1 through 9)
          and a Lambertian BRDF with reflectance :math:`\rho = 0.5`.
        - Illumination:
            - ``directional``: a directional light source at the zenith with
              radiant illumination :math:`L_\mathrm{i} = 1.0`.
            - ``constant``: an isotropic illumination
              :math:`L_\mathrm{i} \in [0.1, 1, 10]`.
        - Sensor: A series of distant directional sensors at
          :math:`\mathrm{VZA} \in [0, \pi/2]` and :math:`\mathrm{VAA} \in [0, 2\pi]`.

    Expected behaviour
    ------------------

    For all scene sizes below the parametrized size :code:`min_expected_size`
    the computational results must be equal to the theoretical prediction within
    a relative tolerance of 1e-3.
    """
    from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f
    from eradiate.scenes.core import KernelDict
    from eradiate.solvers.onedim.runner import OneDimRunner

    min_expected_size = 1e3
    results = dict()
    spp = 1
    zeniths = np.arange(0., 90., 10.)
    azimuths = np.arange(0., 360., 10.)

    expected = np.zeros((len(zeniths), len(azimuths)))
    for i, zenith in enumerate(zeniths):
        expected[i, :] = np.cos(np.deg2rad(zenith)) / (2.0 * np.pi)

    for scene_size in [10 ** i for i in range(1, 9)]:
        result = np.zeros(expected.shape)

        for i, zenith in enumerate(zeniths):
            for j, azimuth in enumerate(azimuths):
                kernel_dict = KernelDict({
                    "type": "scene",
                    "bsdf_surface": {
                        "type": "diffuse",
                        "reflectance": {"type": "uniform", "value": 0.5},
                    },
                    "surface": {
                        "type": "rectangle",
                        "to_world": ScalarTransform4f.scale(
                            ScalarVector3f(scene_size, scene_size, 1)
                        ),
                        "bsdf": {"type": "ref", "id": "bsdf_surface"},
                    },
                    "illumination": {
                        "type": "directional",
                        "direction": [0, 0, -1],
                        "irradiance": {"type": "uniform", "value": 1.0},
                    },
                    "measure": {
                        "type": "distant",
                        "id": "measure",
                        "direction": list(-angles_to_direction(
                            theta=np.deg2rad(zenith),
                            phi=np.deg2rad(azimuth)
                        )),
                        "ray_target": [0, 0, 0],
                        "sampler": {
                            "type": "independent",
                            "sample_count": spp
                        },
                        "film": {
                            "type": "hdrfilm",
                            "width": 1,
                            "height": 1,
                            "pixel_format": "luminance",
                            "component_format": "float32",
                            "rfilter": {"type": "box"}
                        }
                    },
                    "integrator": {"type": "path"},
                })

                solver = OneDimRunner(kernel_dict)

                result[i][j] = solver.run()["measure"].squeeze()

        results[scene_size] = np.allclose(result, expected, rtol=1e-3)

    assert np.all(
        [
            result
            for result in [
                results[size] for size in results if size <= min_expected_size
            ]
        ]
    )

    # write the maximum scene size that passes the test to the benchmarks rst file

    passed_sizes = [size for size in results if results[size]]
    maxsize = np.max(passed_sizes)

    json_metadata["metrics"] = {
        "test_onedimsolver_large_size": {
            "name": "OneDimSolver maximum scene size",
            "description": "The maximum size for OneDimSolver scenes is:",
            "value": f"{float(maxsize):1.1e}",
            "units": "length units",
        }
    }
