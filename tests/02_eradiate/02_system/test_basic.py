"""A series of basic one-dimensional test cases."""

from contextlib import nullcontext

import numpy as np
import pytest

import eradiate
import eradiate.scenes as ertsc
from eradiate import ModeFlags
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.experiments import mitsuba_run
from eradiate.scenes.core import KernelDict


@pytest.mark.parametrize(
    "illumination, spp", [("directional", 1), ("constant", 300000)]
)
@pytest.mark.parametrize("li", [0.1, 1.0, 10.0])
@pytest.mark.slow
def test_radiometric_accuracy(modes_all_mono, illumination, spp, li, ert_seed_state):
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

    * ``directional``: :math:`L_\mathrm{o} = \frac{\rho L_\mathrm{i}}{\pi}`
    * ``constant``: :math:`L_\mathrm{o} = \rho L_\mathrm{i}`
    """

    # Basic configuration
    vza = np.linspace(0, 80, 10)
    rho = 0.5

    # Ignore warning in single precision
    with pytest.warns(UserWarning) if (
        eradiate.mode().has_flags(ModeFlags.MI_SINGLE) and spp > 100000
    ) else nullcontext():
        measure = ertsc.measure.MultiDistantMeasure.from_viewing_angles(
            zeniths=vza, azimuths=0.0, target=[0, 0, 0], spp=spp
        )

    elements = [
        ertsc.surface.BasicSurface(
            bsdf=ertsc.bsdfs.LambertianBSDF(reflectance=rho),
            shape=ertsc.shapes.RectangleShape(edges=2.0 * ureg.m),
        ),
        measure,
        ertsc.integrators.PathIntegrator(),
    ]

    if illumination == "directional":
        elements.append(
            ertsc.illumination.DirectionalIllumination(zenith=0.0, irradiance=li)
        )
        theoretical_solution = np.full_like(vza, rho * li / np.pi)

    elif illumination == "constant":
        elements.append(ertsc.illumination.ConstantIllumination(radiance=li))
        theoretical_solution = np.full_like(vza, rho * li)

    else:
        raise ValueError(f"unsupported illumination '{illumination}'")

    ctx = KernelDictContext()
    kernel_dict = KernelDict.from_elements(*elements, ctx=ctx)
    result = np.array(
        mitsuba_run(kernel_dict, seed_state=ert_seed_state)["values"]["measure"]
    )
    assert np.allclose(result, theoretical_solution, rtol=1e-3)
