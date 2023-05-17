import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.scenes.core import traverse
from eradiate.scenes.shapes import RectangleShape
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs, expected_reflectance",
    [({}, None), ({"bsdf": {"type": "lambertian", "reflectance": 0.0}}, 0.0)],
    ids=["noargs", "args"],
)
def test_rectangle_construct_kernel_dict(modes_all, kwargs, expected_reflectance):
    rectangle = RectangleShape(**kwargs)
    mi_wrapper = check_scene_element(rectangle, mi.Shape)

    if expected_reflectance is not None:
        assert mi_wrapper.parameters["bsdf.reflectance.value"] == expected_reflectance


@pytest.mark.parametrize(
    "kwargs, expected_transform",
    [
        (
            {"edges": [2, 4]},
            [(-1, -2, 0), (1, 2, 0)],
        ),
        (
            {"edges": [2, 2], "center": [0, 0, 1]},
            [(-1, -1, 1), (1, 1, 1)],
        ),
        (
            {"edges": [2, 2], "up": [-1, 0, 0]},
            [(1, -1, 0), (-1, 1, 0)],
        ),
        (
            {"edges": [2, 2], "normal": [0, -1, 0], "up": [-1, 0, 0]},
            [(1, 0, -1), (-1, 0, 1)],
        ),
        (
            {
                "edges": [2, 4],
                "normal": [0, -1, 0],
                "up": [-1, 0, 0],
                "center": [0, 0, 1],
            },
            [(2, 0, 0), (-2, 0, 2)],
        ),
    ],
    ids=["edges", "center", "up", "normal_up", "full"],
)
def test_rectangle_params(mode_mono_double, kwargs, expected_transform):
    # Set edges
    rectangle = RectangleShape(**kwargs)
    template, _ = traverse(rectangle)
    kernel_dict = template.render(ctx=KernelContext())
    to_world = kernel_dict["to_world"]
    assert dr.allclose(
        to_world.transform_affine(mi.Point3f(-1, -1, 0)), expected_transform[0]
    )
    assert dr.allclose(
        to_world.transform_affine(mi.Point3f(1, 1, 0)), expected_transform[1]
    )


def test_rectangle_surface():
    # Check that parameters are correctly specified
    with ucc.override(length="km"):
        surface = RectangleShape.surface(altitude=1.0, width=100.0)
        assert np.allclose(surface.center, [0, 0, 1] * ureg.km)
        assert np.allclose(surface.edges, [100, 100] * ureg.km)
