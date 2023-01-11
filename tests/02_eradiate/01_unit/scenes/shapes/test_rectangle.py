import drjit as dr
import mitsuba as mi
import numpy as np

from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.shapes import RectangleShape


def test_rectangle_construct(mode_mono_double):
    ctx = KernelDictContext()

    # Construct without parameter
    rectangle = RectangleShape()
    kernel_dict = rectangle.kernel_dict(ctx)
    # No BSDF is specified
    assert "bsdf" not in kernel_dict[rectangle.id]
    # But despite that, Mitsuba can create a Shape with a default BSDF
    assert isinstance(kernel_dict.load(), mi.Shape)

    # Set BSDF
    rectangle = RectangleShape(bsdf={"type": "black"})
    kernel_dict = rectangle.kernel_dict(ctx)
    # We have a proper BSDF specification
    assert kernel_dict[rectangle.id]["bsdf"] == {
        "type": "diffuse",
        "reflectance": {"type": "uniform", "value": 0.0},
    }
    assert isinstance(kernel_dict.load(), mi.Shape)


def test_rectangle_params(mode_mono_double):
    ctx = KernelDictContext()

    # Set edges
    rectangle = RectangleShape(edges=[2, 4])
    kernel_dict = rectangle.kernel_dict(ctx)
    to_world = kernel_dict[rectangle.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, 0)), [-1, -2, 0]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 0)), [1, 2, 0])

    # Set center
    rectangle = RectangleShape(edges=[2, 2], center=[0, 0, 1])
    kernel_dict = rectangle.kernel_dict(ctx)
    to_world = kernel_dict[rectangle.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, 0)), [-1, -1, 1]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 0)), [1, 1, 1])

    # Set up
    rectangle = RectangleShape(edges=[2, 2], normal=[0, 0, 1], up=[-1, 0, 0])
    kernel_dict = rectangle.kernel_dict(ctx)
    to_world = kernel_dict[rectangle.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, 0)), [1, -1, 0]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 0)), [-1, 1, 0])

    # Set normal and up
    rectangle = RectangleShape(edges=[2, 2], normal=[0, -1, 0], up=[-1, 0, 0])
    kernel_dict = rectangle.kernel_dict(ctx)
    to_world = kernel_dict[rectangle.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, 0)), [1, 0, -1]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 0)), [-1, 0, 1])

    # Full setup
    rectangle = RectangleShape(
        edges=[2, 4], normal=[0, -1, 0], up=[-1, 0, 0], center=[0, 0, 1]
    )
    kernel_dict = rectangle.kernel_dict(ctx)
    to_world = kernel_dict[rectangle.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, 0)), [2, 0, 0]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 0)), [-2, 0, 2])


def test_rectangle_surface():
    # Check that parameters are correctly specified
    with ucc.override(length="km"):
        surface = RectangleShape.surface(altitude=1.0, width=100.0)
        assert np.allclose(surface.center, [0, 0, 1] * ureg.km)
        assert np.allclose(surface.edges, [100, 100] * ureg.km)
