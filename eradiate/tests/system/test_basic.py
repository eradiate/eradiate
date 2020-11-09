"""A series of basic one-dimensional test cases."""

import numpy as np
import pytest


@pytest.mark.parametrize("illumination,spp", [("directional", 1),
                                              ("constant", 256000)])
@pytest.mark.parametrize("li", [0.1, 1.0, 10.0])
@pytest.mark.slow
def test_radiometric_accuracy(variant_scalar_mono, illumination, spp, li):
    r"""
    Radiometric check (``path``)
    ============================

    This simple test case compares simulated leaving radiance at a Lambertian
    surface with theoretical values.

    Rationale
    ---------

        - Geometry: a square surface with unit size and a Lambertian BRDF with 
          reflectance :math:`\rho = 0.5`. 
        - Illumination: 
            - ``directional``: a directional light source at the zenith with 
              radiant illumination :math:`L_\mathrm{i} \in [0.1, 1, 10]`.
            - ``constant``: an isotropic illumination 
              :math:`L_\mathrm{i} \in [0.1, 1, 10]`. 
        - Sensor: A series of distant directional sensors at 
          :math:`\mathrm{VZA} \in [0, \pi/2]` and :math:`\mathrm{VAA} = 0`.


    Expected behaviour
    ------------------

        The computed solution is equal to the theoretical solution (relative 
        tolerance of 0.1%).

        Theoretical solutions: 

        - `directional`: :math:`L_\mathrm{o} = \frac{\rho L_\mathrm{i}}{\pi}`
        - `constant`: :math:`L_\mathrm{o} = \rho L_\mathrm{i}`

    """
    from eradiate.solvers.onedim.runner import OneDimRunner

    # Basic configuration
    vza = np.linspace(0, 80, 10)
    rho = 0.5

    directions = ", ".join(
        ", ".join(str(y) for y in x) for x in [[-np.sin(theta), -np.sin(theta), -np.cos(theta)]
                                               for theta in np.deg2rad(vza)]
    )
    origins = ", ".join(str(x) for x in [0, 0, 0.01] * len(vza))

    solver = OneDimRunner()
    solver.kernel_dict["surface"] = {
        "type": "rectangle",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": rho}
        }
    }

    solver.kernel_dict["measure"] = {
        "type": "radiancemeterarray",
        "origins": origins,
        "directions": directions,
        "id": "measure",
        "sampler": {
            "type": "independent",
            "sample_count": spp
        },
        "film": {
            "type": "hdrfilm",
            "width": len(vza),
            "height": 1,
            "pixel_format": "luminance",
            "component_format": "float32",
            "rfilter": {"type": "box"}
        }
    }

    if illumination == "directional":
        solver.kernel_dict["illumination"] = {
            "type": "directional",
            "direction": [0, 0, -1],
            "irradiance": {"type": "uniform", "value": li}
        }
        theoretical_solution = np.full_like(vza, rho * li / np.pi)

    elif illumination == "constant":
        solver.kernel_dict["illumination"] = {
            "type": "constant",
            "radiance": {"type": "uniform", "value": li}
        }
        theoretical_solution = np.full_like(vza, rho * li)

    result = solver.run()["measure"]
    assert np.allclose(result, theoretical_solution, rtol=1e-3)
