import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.scenes.core import traverse
from eradiate.scenes.shapes import SphereShape
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs, expected_reflectance",
    [({}, None), ({"bsdf": {"type": "lambertian", "reflectance": 0.0}}, 0.0)],
    ids=["noargs", "args"],
)
def test_sphere_construct_kernel_dict(modes_all, kwargs, expected_reflectance):
    sphere = SphereShape(**kwargs)
    mi_wrapper = check_scene_element(sphere, mi.Shape)

    if expected_reflectance is not None:
        assert mi_wrapper.parameters["bsdf.reflectance.value"] == expected_reflectance


def test_sphere_construct_trafo(modes_all):
    assert SphereShape(to_world=mi.Transform4f.scale(2))


def test_sphere_template(modes_all):
    sphere = SphereShape(center=(1, 1, 1), radius=2)

    template, _ = traverse(sphere)
    kernel_dict = template.render(ctx=KernelContext())

    assert kernel_dict["type"] == "sphere"
    assert np.allclose(kernel_dict["center"], (1, 1, 1))
    assert kernel_dict["radius"] == 2

    assert "to_world" not in kernel_dict

    sphere = SphereShape(
        center=(1, 1, 1), radius=2, to_world=mi.ScalarTransform4f.translate((1, 0, 0))
    )
    template, _ = traverse(sphere)
    kernel_dict = template.render(ctx=KernelContext())

    assert np.allclose(kernel_dict["center"], (1, 1, 1))
    assert kernel_dict["radius"] == 2
    assert kernel_dict["to_world"] == mi.ScalarTransform4f.translate((1, 0, 0))


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


def test_sphere_bbox():
    sphere = SphereShape(center=[1, 1, 1], radius=2.0)
    bbox = sphere.bbox
    np.testing.assert_array_equal(bbox.min.m_as(ureg.m), [-1, -1, -1])
    np.testing.assert_array_equal(bbox.max.m_as(ureg.m), [3, 3, 3])

    sphere = SphereShape(center=[1, 1, 1], radius=2.0, to_world=mi.Transform4f.scale(2))
    bbox = sphere.bbox
    np.testing.assert_array_equal(bbox.min.m_as(ureg.m), [-2, -2, -2])
    np.testing.assert_array_equal(bbox.max.m_as(ureg.m), [6, 6, 6])
