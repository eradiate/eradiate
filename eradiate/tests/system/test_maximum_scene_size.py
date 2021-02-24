import numpy as np

from eradiate.frame import angles_to_direction
from eradiate.solvers.core import runner


def test_maximum_scene_size(mode_mono_double, json_metadata):
    r"""
    Maximum scene size (``path``)
    =============================

    This test searches for an order of magnitude of the maximum size a scene using
    a ``distant`` sensor without ray origin control can have. An arbitrary threshold
    is used as the pass/fail criterion for regression control.

    Rationale
    ---------

    - Geometry: a square surface with increasing sizes from 1.0 to 1e9.
      and a Lambertian BRDF with reflectance :math:`\rho = 0.5`.
    - Illumination: a directional light source at the zenith with radiance
      :math:`L_\mathrm{i} = 1.0`.
    - Sensor: a ``distant`` sensor targeting (0, 0, 0) with default ray origin control.

    Expected behaviour
    ------------------

    For all scene sizes below the parametrized size :code:`min_expected_size`
    the computational results must be equal to the theoretical prediction within
    a relative tolerance of 1e-5.
    """
    from eradiate.kernel.core import (
        ScalarTransform4f,
        ScalarVector3f
    )

    min_expected_size = 1e2
    results = dict()
    spp = 1
    rho = 0.5
    li = 1.0
    expected = rho * li / np.pi

    for scene_size in sorted(
        [10.0 ** i for i in range(1, 9)]
        + [2.0 * 10 ** i for i in range(1, 8)]
        + [5.0 * 10 ** i for i in range(1, 8)]
    ):
        kernel_dict = {
            "type": "scene",
            "bsdf_surface": {
                "type": "diffuse",
                "reflectance": rho,
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
                "irradiance": li,
            },
            "measure": {
                "type": "distant",
                "id": "measure",
                "ray_target": [0, 0, 0],
                "sampler": {"type": "independent", "sample_count": spp},
                "film": {
                    "type": "hdrfilm",
                    "width": 32,
                    "height": 32,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"},
                },
            },
            "integrator": {"type": "path"},
        }

        result = runner(kernel_dict)["measure"].squeeze()
        results[scene_size] = np.allclose(result, expected, rtol=1e-5)

    # Report test metrics
    passed_sizes = [size for size in results if results[size]]
    maxsize = np.max(passed_sizes)

    json_metadata["metrics"] = {
        "test_maximum_scene_size": {
            "name": "Maximum scene size",
            "description": "The maximum scene size is:",
            "value": f"{float(maxsize):1.1e}",
            "units": "length units",
        }
    }

    # Final assertion
    assert np.all(
        [
            result
            for result in [
                results[size] for size in results if size <= min_expected_size
            ]
        ]
    )
