import mitsuba as mi
import numpy as np

from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.shapes import SphereShape


def test_sphere_construct(mode_mono_double):
    ctx = KernelDictContext()

    # Construct without parameter
    sphere = SphereShape()
    kernel_dict = sphere.kernel_dict(ctx)
    # No BSDF is specified
    assert "bsdf" not in kernel_dict[sphere.id]
    # But despite that, Mitsuba can create a Shape with a default BSDF
    assert isinstance(kernel_dict.load(), mi.Shape)

    # Set BSDF
    sphere = SphereShape(bsdf={"type": "black"})
    kernel_dict = sphere.kernel_dict(ctx)
    # We have a proper BSDF specification
    assert kernel_dict[sphere.id]["bsdf"] == {
        "type": "diffuse",
        "reflectance": {"type": "uniform", "value": 0.0},
    }
    assert isinstance(kernel_dict.load(), mi.Shape)


def test_sphere_surface():
    # Check that parameters are correctly specified
    with ucc.override(length="km"):
        surface = SphereShape.surface(altitude=1.0, planet_radius=1.0)
        assert np.allclose(surface.center, [0, 0, -1] * ureg.km)
        assert np.allclose(surface.radius, 2.0 * ureg.km)


def test_sphere_atmosphere():
    # Check that parameters are correctly specified
    with ucc.override(length="km"):
        sphere = SphereShape.atmosphere(top=1.0, planet_radius=1.0)
        assert np.allclose(sphere.center, [0, 0, -1] * ureg.km)
        assert np.allclose(sphere.radius, 2.0 * ureg.km)


def test_sphere_contains():
    sphere = SphereShape(center=[1, 1, 1], radius=2.0)

    # Works with a single point
    assert sphere.contains([1.5, 1.5, 1.5])
    assert not sphere.contains([1.5, 2.0, -2.0])
    assert not sphere.contains([1.5, 1.5, 1.5] * ureg.km)

    # Works with multiple points
    assert np.all(sphere.contains([[1.5, 1.5, 1.5], [1.5, 2.0, -2.0]]) == [True, False])
