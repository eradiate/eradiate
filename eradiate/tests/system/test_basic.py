"""A series of basic one-dimensional test cases."""

import numpy as np
import pytest

from eradiate.solvers.core import runner


@pytest.mark.parametrize("illumination,spp", [("directional", 1), ("constant", 256000)])
@pytest.mark.parametrize("li", [0.1, 1.0, 10.0])
@pytest.mark.slow
def test_radiometric_accuracy(mode_mono, illumination, spp, li):
    r"""
    Radiometric check (``path``)
    ============================

    This simple test case compares simulated leaving radiance at a Lambertian
    surface with theoretical values.

    Rationale
    ---------

    * Geometry: a square surface with unit size and a Lambertian BRDF with
      reflectance :math:`\rho = 0.5`.
    * Illumination:

      * ``directional``: a directional light source at the zenith with
        radiance :math:`L_\mathrm{i} \in [0.1, 1, 10]`.
      * ``constant``: an isotropic illumination with radiance
        :math:`L_\mathrm{i} \in [0.1, 1, 10]`.

    * Sensor: A ``distant`` sensor targeting (0, 0, 0).


    Expected behaviour
    ------------------

    The computed solution is equal to the theoretical solution (relative
    tolerance of 0.1%).

    Theoretical solutions:

    * `directional`: :math:`L_\mathrm{o} = \frac{\rho L_\mathrm{i}}{\pi}`
    * `constant`: :math:`L_\mathrm{o} = \rho L_\mathrm{i}`
    """

    # Basic configuration
    vza = np.linspace(0, 80, 10)
    rho = 0.5

    kernel_dict = {
        "type": "scene",
        "surface": {
            "type": "rectangle",
            "bsdf": {
                "type": "diffuse",
                "reflectance": rho,
            },
        },
        "measure": {
            "type": "distant",
            "id": "measure",
            "ray_target": [0, 0, 0],
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {
                "type": "hdrfilm",
                "width": len(vza),
                "height": 1,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        },
        "integrator": {"type": "path"},
    }

    if illumination == "directional":
        kernel_dict["illumination"] = {
            "type": "directional",
            "direction": [0, 0, -1],
            "irradiance": {"type": "uniform", "value": li},
        }
        theoretical_solution = np.full_like(vza, rho * li / np.pi)

    elif illumination == "constant":
        kernel_dict["illumination"] = {
            "type": "constant",
            "radiance": {"type": "uniform", "value": li},
        }
        theoretical_solution = np.full_like(vza, rho * li)

    else:
        raise ValueError(f"unsupported illumination '{illumination}'")

    result = runner(kernel_dict)["measure"]
    assert np.allclose(result, theoretical_solution, rtol=1e-3)
