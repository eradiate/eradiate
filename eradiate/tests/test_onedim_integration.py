import numpy as np
import pytest

@pytest.mark.slow
def test_onedimsolver_large_size(variant_scalar_mono_double, json_metadata):
    r"""
    Maximum scene size (``path``)
    -----------------------------

    This test case asserts that the maximum scene size that can be rendered without
    breaking the limits of numerical precision is above a pre set limit.


    Rationale
    ^^^^^^^^^

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
    ^^^^^^^^^^^^^^^^^^

    For all scene sizes below the parametrized size :code:`min_expected_size` the computational
    results must be equal to the theoretical prediction within a relative tolerance of 1e-3.
    """
    from eradiate.scenes import SceneDict
    from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f
    from eradiate.solvers.onedim import OneDimSolver

    min_expected_size = 1e3
    results = dict()
    for scene_size in [10 ** i for i in range(1, 9)]:

        scene_dict = SceneDict(
            {
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
                "integrator": {"type": "path"},
            }
        )

        solver = OneDimSolver(scene_dict)

        vza = np.linspace(0, 90, 10)
        vaa = np.linspace(0, 360, 37)

        result = solver.run(vza=vza, vaa=vaa, spp=1024)

        results[scene_size] = np.allclose(result, 1.0 / (2.0 * np.pi), rtol=1e-3)

    assert np.all(
        [
            result
            for result in [
                results[size] for size in results if size <= min_expected_size
            ]
        ]
    )

    # write the maximum scene size that passes the test to the benchmarks rst file

    passed_sizes = [size for size in results if results[size] == True]
    maxsize = np.max(passed_sizes)

    json_metadata['metrics'] = {
        "test_onedimsolver_large_size": {
            "name": "OneDimSolver maximum scene size",
            "description": "The maximum size for OneDimSolver scenes is:",
            "value": f"{float(maxsize):1.1e}",
            "unit": "length units",
        }
    }
