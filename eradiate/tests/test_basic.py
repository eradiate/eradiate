"""A series of basic one-dimensional test cases."""

import numpy as np

import attr
import eradiate
import pytest
from eradiate.scenes.builder import *


@pytest.mark.parametrize("illumination", ["directional", "constant"])
@pytest.mark.parametrize("li", [0.1, 1.0, 10.0])
def test_radiometric_accuracy(variant_scalar_mono, illumination, li):
    r"""
    Radiometric check (``path``)
    ----------------------------

    This simple test case compares simulated leaving radiance at a Lambertian
    surface with theoretical values.


    Rationale
    ^^^^^^^^^

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
    ^^^^^^^^^^^^^^^^^^

        The computed solution is equal to the theoretical solution (relative 
        tolerance of 0.1%).

        Theoretical solutions: 
        
        - `directional`: :math:`L_\mathrm{o} = \frac{\rho L_\mathrm{i}}{\pi}`
        - `constant`: :math:`L_\mathrm{o} = \rho L_\mathrm{i}`

    """
    from eradiate.solvers.onedim import OneDimSolver

    # Basic configuration
    vza = np.linspace(0, 90, 11)
    rho = 0.5

    solver = OneDimSolver()
    solver.scene.bsdfs = [
        bsdfs.Diffuse(id="brdf_surface", reflectance=Spectrum(rho))
    ]

    if illumination == "directional":
        solver.scene.emitter = \
            emitters.Directional(direction=[0, 0, -1], irradiance=Spectrum(li))
        theoretical_solution = np.full_like(vza, rho * li / np.pi)

    elif illumination == "constant":
        theoretical_solution = np.full_like(vza, rho * li)
        solver.scene.emitter = \
            emitters.Constant(radiance=Spectrum(li))

    result = solver.run(vza=vza, vaa=0., spp=3200)
    assert np.allclose(result, theoretical_solution, rtol=1e-3)
