import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import traverse
from eradiate.scenes.shapes import CuboidShape


@pytest.mark.parametrize(
    "kwargs, expected_reflectance",
    [({}, None), ({"bsdf": {"type": "black"}}, 0.0)],
    ids=["noargs", "args"],
)
def test_cuboid_construct_kernel_dict(modes_all, kwargs, expected_reflectance):
    cuboid = CuboidShape(**kwargs)
    template, _ = traverse(cuboid)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx=ctx)

    if expected_reflectance is not None:
        assert kernel_dict["bsdf"]["reflectance"]["value"] == expected_reflectance
    assert isinstance(mi.load_dict(kernel_dict), mi.Shape)


@pytest.mark.parametrize(
    "kwargs, expected_transform",
    [
        (
            {"edges": [2, 4, 8]},
            [(-1, -2, -4), (1, 2, 4)],
        ),
        (
            {"edges": [2, 2, 2], "center": [1, 1, 1]},
            [(0, 0, 0), (2, 2, 2)],
        ),
    ],
    ids=["edges", "center"],
)
def test_cuboid_params(mode_mono_double, kwargs, expected_transform):
    # Set edges
    cuboid = CuboidShape(**kwargs)
    template, _ = traverse(cuboid)
    kernel_dict = template.render(ctx=KernelDictContext())
    to_world = kernel_dict["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.Point3f(-1, -1, -1)), expected_transform[0]
    )
    assert dr.allclose(
        to_world.transform_affine(mi.Point3f(1, 1, 1)), expected_transform[1]
    )


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
        template, _ = traverse(cuboid)
        kernel_dict = template.render(ctx=KernelDictContext())
        bbox = mi.load_dict(kernel_dict).bbox()
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
