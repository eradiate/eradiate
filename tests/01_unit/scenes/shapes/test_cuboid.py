import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.core import traverse
from eradiate.scenes.shapes import CuboidShape
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs, expected_reflectance",
    [({}, None), ({"bsdf": {"type": "lambertian", "reflectance": 0.0}}, 0.0)],
    ids=["noargs", "args"],
)
def test_cuboid_construct_kernel_dict(modes_all, kwargs, expected_reflectance):
    cuboid = CuboidShape(**kwargs)

    mi_wrapper = check_scene_element(cuboid, mi.Shape)

    if expected_reflectance is not None:
        assert mi_wrapper.parameters["bsdf.reflectance.value"] == expected_reflectance


def test_cuboid_construct_trafo(mode_mono):
    assert CuboidShape(
        edges=[1, 1, 1],
        to_world=mi.ScalarTransform4f().rotate(axis=[1, 1, 1], angle=45),
    )


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
        ({"to_world": True}, [(1, 1, 1), (3, 3, 3)]),
        (
            {"edges": [2, 2, 2], "center": [1, 1, 1], "to_world": True},
            [(1, 1, 1), (3, 3, 3)],
        ),
    ],
    ids=["edges", "center", "to_world", "full"],
)
def test_cuboid_params(mode_mono_double, kwargs, expected_transform):
    if "to_world" in kwargs:
        trafo = kwargs.pop("to_world")
        if trafo:
            to_world = mi.ScalarTransform4f().translate((2, 2, 2))
            cuboid = CuboidShape(**kwargs, to_world=to_world)
    else:
        cuboid = CuboidShape(**kwargs)

    template, _ = traverse(cuboid)
    kernel_dict = template.render(ctx=KernelContext())
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
        kernel_dict = template.render(ctx=KernelContext())
        bbox = mi.load_dict(kernel_dict).bbox()
        assert dr.allclose(bbox.min, [-500, -500, -500])
        assert dr.allclose(bbox.max, [500, 500, 1000])


def test_cuboid_contains(mode_mono):
    cuboid = CuboidShape(center=[0.5, 0.5, 0.5], edges=[1, 1, 1])

    # Works with a single point
    assert cuboid.contains([0.5, 0.5, 0.5])
    assert not cuboid.contains([0.5, 0.5, -0.5])
    assert not cuboid.contains([0.5, 0.5, 0.5] * ureg.km)

    # Works with multiple points
    assert np.all(cuboid.contains([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]) == [True, False])

    # to_world
    import mitsuba as mi

    cuboid = CuboidShape(
        to_world=mi.ScalarTransform4f().rotate(angle=45, axis=(0, 0, 1))
    )

    # Works with a single point
    assert cuboid.contains([0.5, 0.5, 0.5])
    assert not cuboid.contains([1.0, 1.0, 1.0])
    assert not cuboid.contains([0.5, 0.5, 0.5] * ureg.km)

    # Works with multiple points
    assert np.all(cuboid.contains([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]) == [True, False])


def test_cuboid_bbox(mode_mono):
    cuboid = CuboidShape(center=[0.5, 0.5, 0.5], edges=[1, 1, 1])
    bbox = cuboid.bbox
    np.testing.assert_array_equal(bbox.min.m_as(ureg.m), [0, 0, 0])
    np.testing.assert_array_equal(bbox.max.m_as(ureg.m), [1, 1, 1])

    import mitsuba as mi

    cuboid = CuboidShape(to_world=mi.ScalarTransform4f().translate((1, 2, 3)))
    bbox = cuboid.bbox
    np.testing.assert_array_equal(bbox.min.m_as(ureg.m), [0, 1, 2])
    np.testing.assert_array_equal(bbox.max.m_as(ureg.m), [2, 3, 4])
