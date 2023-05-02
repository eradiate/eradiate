import mitsuba as mi
import numpy as np

import eradiate
from eradiate import KernelContext
from eradiate.kernel import mi_render, mi_traverse


def test_maximum_scene_size(modes_all_mono, json_metadata):
    r"""
    Maximum scene size (``path``)
    =============================

    This test searches for an order of magnitude of the maximum size a scene using
    a ``hdistant`` sensor without ray origin control can have. An arbitrary
    threshold is used as the pass/fail criterion for regression control.

    Rationale
    ---------

    - Geometry: a square surface with increasing sizes from 1.0 to 1e9.
      and a Lambertian BRDF with reflectance :math:`\rho = 0.5`.
    - Illumination: a directional light source at the zenith with radiance
      :math:`L_\mathrm{i} = 1.0`.
    - Sensor: a ``hdistant`` sensor targeting (0, 0, 0).

    Expected behaviour
    ------------------

    For all scene sizes below the parametrized size :code:`expected_min_size`
    the computational results must be equal to the theoretical prediction.
    Tolerance is set according to the defaults for :func:`numpy.allclose`.
    Metrics are reported only for the double precision version of this test.
    """
    expected_min_size = 1e8 if eradiate.mode().is_single_precision else 1e12
    spp = 1
    rho = 0.5
    li = 1.0
    expected = rho * li / np.pi
    max_i = 12
    scene_sizes = np.array(
        sorted(
            [10.0**i for i in range(1, max_i)]
            + [2.0 * 10**i for i in range(1, max_i)]
            + [5.0 * 10**i for i in range(1, max_i)]
            + [10.0**max_i]
        )
    )
    passed = np.full_like(scene_sizes, False, dtype=bool)

    for i, scene_size in enumerate(scene_sizes):
        mi_wrapper = mi_traverse(
            mi.load_dict(
                {
                    "type": "scene",
                    "bsdf_surface": {
                        "type": "diffuse",
                        "reflectance": rho,
                    },
                    "surface": {
                        "type": "rectangle",
                        "to_world": mi.ScalarTransform4f.scale(
                            mi.ScalarVector3f(scene_size, scene_size, 1)
                        ),
                        "bsdf": {"type": "ref", "id": "bsdf_surface"},
                    },
                    "illumination": {
                        "type": "directional",
                        "direction": [0, 0, -1],
                        "irradiance": li,
                    },
                    "measure": {
                        "type": "hdistant",
                        "id": "measure",
                        "target": [0, 0, 0],
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
            )
        )

        result = np.squeeze(
            mi_render(mi_wrapper, ctxs=[KernelContext()])[550.0]["measure"]
        )
        passed[i] = np.allclose(result, expected)

    # Report test metrics
    max_size = float(np.max(scene_sizes[passed])) if np.any(passed) else 0.0

    if eradiate.mode().is_double_precision:
        json_metadata["metrics"] = {
            "test_maximum_scene_size": {
                "name": "Maximum scene size",
                "description": "The maximum scene size is: ",
                "value": f"{max_size:1.1e}",
                "units": "length units",
            }
        }

    # Maximum size for which result is correct is greater than requested min size
    assert max_size >= expected_min_size
    # All sizes below the threshold produce correct results
    assert np.all(passed[scene_sizes <= expected_min_size])
