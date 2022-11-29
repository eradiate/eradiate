import mitsuba
import pytest
from rich.pretty import pprint

from eradiate.contexts import KernelDictContext
from eradiate.exceptions import TraversalError
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.shapes import RectangleShape, Shape
from eradiate.scenes.surface import BasicSurface


@pytest.mark.parametrize(
    "kwargs, expected_shape, expected_traversal",
    [
        ({}, None, TraversalError),
        ({"shape": {"type": "rectangle"}}, RectangleShape, {"surface_bsdf": "nope"}),
    ],
    ids=[
        "noargs",
        "shape",
    ],
)
def test_basic_surface_construct(
    modes_all_double, kwargs, expected_shape, expected_traversal
):
    surface = BasicSurface(**kwargs)

    if expected_shape is None:
        assert surface.shape is None
    elif issubclass(expected_shape, Shape):
        assert isinstance(surface.shape, expected_shape)
    else:
        raise NotImplementedError

    if isinstance(expected_traversal, dict):
        template, params = traverse(surface)
        print(template.render(KernelDictContext()))

        scene = Scene(objects={"surface": surface})
        template, params = traverse(scene)
        kernel_dict = template.render(KernelDictContext())
        pprint(kernel_dict)
        mitsuba.load_dict(kernel_dict)

    elif issubclass(expected_traversal, TraversalError):
        with pytest.raises(expected_traversal):
            traverse(surface)
    else:
        raise NotImplementedError

    # with pytest.raises(ValueError):
    #     surface.kernel_dict(ctx).load()
    #
    # # Specify shape
    # surface = BasicSurface(shape={"type": "rectangle"})
    # kernel_dict = surface.kernel_dict(ctx)
    # # The BSDF is referenced
    # assert kernel_dict.data[surface.shape_id]["bsdf"]["type"] == "ref"
    # # The constructed dict can be loaded
    # assert kernel_dict.load()
