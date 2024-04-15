import mitsuba as mi
import pytest

from eradiate import KernelContext
from eradiate.exceptions import TraversalError
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.shapes import RectangleShape, Shape
from eradiate.scenes.surface import BasicSurface
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs, expected_shape, expected_traversal_param_keys",
    [
        ({}, None, TraversalError),
        (
            {"shape": {"type": "rectangle"}},
            RectangleShape,
            {"surface_bsdf.reflectance.value"},
        ),
    ],
    ids=[
        "noargs",
        "shape",
    ],
)
def test_basic_surface_construct(
    modes_all_double, kwargs, expected_shape, expected_traversal_param_keys
):
    surface = BasicSurface(**kwargs)

    if expected_shape is None:
        assert surface.shape is None
    elif issubclass(expected_shape, Shape):
        assert isinstance(surface.shape, expected_shape)
    else:
        raise NotImplementedError

    if isinstance(expected_traversal_param_keys, set):
        template, params = traverse(surface)

        # Scene element is composite: template has not "type" key
        assert "type" not in template
        # Parameter map keys are fetched recursively
        assert set(params.keys()) == expected_traversal_param_keys

        # When enclosed in a Scene, the surface can be traversed
        scene = Scene(objects={"surface": surface})
        template, params = traverse(scene)
        kernel_dict = template.render(KernelContext())
        assert isinstance(mi.load_dict(kernel_dict), mi.Scene)

    elif isinstance(expected_traversal_param_keys, type) and issubclass(
        expected_traversal_param_keys, TraversalError
    ):
        with pytest.raises(
            expected_traversal_param_keys,
            match="A 'BasicSurface' cannot be traversed if its 'shape' field is unset",
        ):
            traverse(surface)
    else:
        raise NotImplementedError


def test_basic_surface_kernel_dict(mode_mono):
    surface = BasicSurface(shape={"type": "rectangle"})
    check_scene_element(surface)
