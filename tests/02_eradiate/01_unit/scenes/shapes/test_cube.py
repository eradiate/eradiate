import drjit as dr
import mitsuba as mi
import numpy as np

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.shapes import CuboidShape


def test_cube_construct(modes_all_double):
    ctx = KernelDictContext()

    # Construct without parameter
    cube = CuboidShape()
    kernel_dict = cube.kernel_dict(ctx)
    # No BSDF is specified
    assert "bsdf" not in kernel_dict[cube.id]
    # But despite that, Mitsuba can create a Shape with a default BSDF
    assert isinstance(kernel_dict.load(), mi.Shape)

    # Set BSDF
    cube = CuboidShape(bsdf={"type": "black"})
    kernel_dict = cube.kernel_dict(ctx)
    # We have a proper BSDF specification
    assert kernel_dict[cube.id]["bsdf"] == {
        "type": "diffuse",
        "reflectance": {"type": "uniform", "value": 0.0},
    }
    assert isinstance(kernel_dict.load(), mi.Shape)


def test_cuboid_params(mode_mono_double):
    ctx = KernelDictContext()

    # Set edges
    cuboid = CuboidShape(edges=[2, 4, 8])
    kernel_dict = cuboid.kernel_dict(ctx)
    to_world = kernel_dict[cuboid.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, -1)), [-1, -2, -4]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 1)), [1, 2, 4])

    # Set centre
    cuboid = CuboidShape(edges=[2, 2, 2], center=[1, 1, 1])
    kernel_dict = cuboid.kernel_dict(ctx)
    to_world = kernel_dict[cuboid.id]["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.ScalarPoint3f(-1, -1, -1)), [0, 0, 0]
    )
    assert dr.allclose(to_world.transform_affine(mi.ScalarPoint3f(1, 1, 1)), [2, 2, 2])


def test_cuboid_atmosphere(mode_mono_double):
    with ucc.override(length="km"):
        cuboid = CuboidShape.atmosphere(
            top=1.0,
            bottom=0.0,
            bottom_offset=-0.5,
            width=1.0,
        )
        assert np.allclose(cuboid.edges, [1, 1, 1.5] * ureg.km)
        assert np.allclose(cuboid.center, [0, 0, 0.25] * ureg.km)

    with uck.override(length="m"):
        ctx = KernelDictContext()
        bbox = cuboid.kernel_dict(ctx).load().bbox()
        assert dr.allclose(bbox.min, [-500, -500, -500])
        assert dr.allclose(bbox.max, [500, 500, 1000])


def test_cuboid_contains():
    cuboid = CuboidShape(center=[0.5, 0.5, 0.5], edges=[1, 1, 1])

    # Works with a single point
    assert cuboid.contains([0.5, 0.5, 0.5])
    assert not cuboid.contains([0.5, 0.5, -0.5])
    assert not cuboid.contains([0.5, 0.5, 0.5] * ureg.km)

    # Works with multiple points
    assert np.all(cuboid.contains([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]) == [True, False])
