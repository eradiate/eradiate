"""A series of basic one-dimensional test cases."""

from contextlib import nullcontext

import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.kernel import mi_render
from eradiate.scenes.bsdfs import LambertianBSDF
from eradiate.scenes.core import Scene
from eradiate.scenes.illumination import (
    AstroObjectIllumination,
    ConstantIllumination,
    DirectionalIllumination,
)
from eradiate.scenes.integrators import PathIntegrator
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.shapes import RectangleShape
from eradiate.scenes.surface import BasicSurface
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "illumination, spp",
    [
        ("directional", 1),
        ("constant", 5e5),
        # Deactivated for now (inexplicably fails when run as part of full test suite)
        # ("astro_object", 5e5),
    ],
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
    with pytest.warns(UserWarning, match="the selected mode is single-precision") if (
        eradiate.mode().is_single_precision and spp > 100000
    ) else nullcontext():
        measure = MultiDistantMeasure.hplane(
            zeniths=vza, azimuth=0.0, target=[0, 0, 0], spp=spp
        )

    objects = {
        "surface": BasicSurface(
            bsdf=LambertianBSDF(reflectance=rho),
            shape=RectangleShape(edges=2.0 * ureg.m),
        ),
        "measure": measure,
        "integrator": PathIntegrator(),
    }

    if illumination == "directional":
        objects["illumination"] = DirectionalIllumination(zenith=0.0, irradiance=li)
        theoretical_solution = np.full_like(vza, rho * li / np.pi)
        rtol = 1e-3

    elif illumination == "constant":
        objects["illumination"] = ConstantIllumination(radiance=li)
        theoretical_solution = np.full_like(vza, rho * li)
        rtol = 1e-3

    # Deactivated for now (see parametrization)
    elif illumination == "astro_object":
        objects["illumination"] = AstroObjectIllumination(
            zenith=0.0, irradiance=li, angular_diameter=0.03
        )
        theoretical_solution = np.full_like(vza, rho * li / np.pi)
        # The angular diameter is not taken into account in the theoretical solution
        rtol = 1e-2

    else:
        raise ValueError(f"unsupported illumination '{illumination}'")

    scene = Scene(objects=objects)
    mi_wrapper = check_scene_element(scene, mi.Scene)

    result = np.squeeze(mi_render(mi_wrapper, ctxs=[KernelContext()])[550.0]["measure"])
    np.testing.assert_allclose(result, theoretical_solution, rtol=rtol)
