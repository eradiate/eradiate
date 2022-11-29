import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.shapes import SphereShape


@pytest.mark.parametrize(
    "kwargs, expected_reflectance",
    [({}, None), ({"bsdf": {"type": "black"}}, 0.0)],
    ids=["noargs", "args"],
)
def test_sphere_construct_kernel_dict(modes_all, kwargs, expected_reflectance):
    sphere = SphereShape(**kwargs)
    template, _ = traverse(sphere)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx=ctx)

    if expected_reflectance is not None:
        assert kernel_dict["bsdf"]["reflectance"]["value"] == expected_reflectance
    assert isinstance(mi.load_dict(kernel_dict), mi.Shape)


def test_sphere_surface():
    # Check that parameters are correctly specified
    with ucc.override(length="km"):
        surface = SphereShape.surface(altitude=1.0, planet_radius=1.0)
        assert np.allclose(surface.center, [0, 0, 0] * ureg.km)
        assert np.allclose(surface.radius, 2.0 * ureg.km)


def test_sphere_atmosphere():
    # Check that parameters are correctly specified
    with ucc.override(length="km"):
        sphere = SphereShape.atmosphere(top=1.1, planet_radius=1.0)
        assert np.allclose(sphere.center, [0, 0, 0] * ureg.km)
        assert np.allclose(sphere.radius, 2.1 * ureg.km)


def test_sphere_contains():
    sphere = SphereShape(center=[1, 1, 1], radius=2.0)

    # Works with a single point
    assert sphere.contains([1.5, 1.5, 1.5])
    assert not sphere.contains([1.5, 2.0, -2.0])
    assert not sphere.contains([1.5, 1.5, 1.5] * ureg.km)

    # Works with multiple points
    assert np.all(sphere.contains([[1.5, 1.5, 1.5], [1.5, 2.0, -2.0]]) == [True, False])
