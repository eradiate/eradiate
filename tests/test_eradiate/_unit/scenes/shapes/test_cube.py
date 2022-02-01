import enoki as ek
import numpy as np

from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.shapes import CuboidShape


def test_cube_construct(modes_all_double):
    from mitsuba.render import Shape

    ctx = KernelDictContext()

    # Construct without parameter
    cube = CuboidShape()
    kernel_dict = cube.kernel_dict(ctx)
    # No BSDF is specified
    assert "bsdf" not in kernel_dict[cube.id]
    # But despite that, Mitsuba can create a Shape with a default BSDF
    assert isinstance(kernel_dict.load(), Shape)

    # Set BSDF
    cube = CuboidShape(bsdf={"type": "black"})
    kernel_dict = cube.kernel_dict(ctx)
    # We have a proper BSDF specification
    assert kernel_dict[cube.id]["bsdf"] == {
        "type": "diffuse",
        "reflectance": {"type": "uniform", "value": 0.0},
    }
    assert isinstance(kernel_dict.load(), Shape)


def test_cuboid_params(mode_mono_double):
    from mitsuba.core import Point3f

    ctx = KernelDictContext()

    # Set edges
    cuboid = CuboidShape(edges=[2, 4, 8])
    kernel_dict = cuboid.kernel_dict(ctx)
    to_world = kernel_dict[cuboid.id]["to_world"]
    assert ek.allclose(to_world.transform_affine(Point3f(-1, -1, -1)), [-1, -2, -4])
    assert ek.allclose(to_world.transform_affine(Point3f(1, 1, 1)), [1, 2, 4])

    # Set centre
    cuboid = CuboidShape(edges=[2, 2, 2], center=[1, 1, 1])
    kernel_dict = cuboid.kernel_dict(ctx)
    to_world = kernel_dict[cuboid.id]["to_world"]
    assert ek.allclose(to_world.transform_affine(Point3f(-1, -1, -1)), [0, 0, 0])
    assert ek.allclose(to_world.transform_affine(Point3f(1, 1, 1)), [2, 2, 2])


def test_cuboid_atmosphere():
    with ucc.override(length="km"):
        cuboid = CuboidShape.atmosphere(
            top=1.0,
            bottom=0.0,
            bottom_offset=-1.0,
            width=1.0,
        )
        assert np.allclose(cuboid.edges, [1, 1, 2] * ureg.km)


def test_cuboid_contains():
    cuboid = CuboidShape(center=[0.5, 0.5, 0.5], edges=[1, 1, 1])

    # Works with a single point
    assert cuboid.contains([0.5, 0.5, 0.5])
    assert not cuboid.contains([0.5, 0.5, -0.5])
    assert not cuboid.contains([0.5, 0.5, 0.5] * ureg.km)

    # Works with multiple points
    assert np.all(cuboid.contains([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]) == [True, False])
