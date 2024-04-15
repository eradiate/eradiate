import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate.exceptions import TraversalError
from eradiate.scenes.core import Ref, Scene, traverse
from eradiate.scenes.shapes import RectangleShape, Shape
from eradiate.scenes.surface import CentralPatchSurface
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize(
    "kwargs, expected_shape, expected_traversal_param_keys",
    [
        ({}, None, TraversalError),
        (
            {"shape": {"type": "rectangle"}},
            RectangleShape,
            {"surface_bsdf.bsdf_0.reflectance.value"},
        ),
        (
            {"shape": {"type": "rectangle"}, "patch_edges": 1.0},
            RectangleShape,
            {"surface_bsdf.bsdf_0.reflectance.value"},
        ),
        (
            {"shape": {"type": "rectangle"}, "patch_edges": [1.0, 0.5]},
            RectangleShape,
            {"surface_bsdf.bsdf_0.reflectance.value"},
        ),
    ],
    ids=[
        "noargs",
        "shape",
        "edges_scalar",
        "edges_vector",
    ],
)
def test_central_patch_construct_new(
    modes_all_double, kwargs, expected_shape, expected_traversal_param_keys
):
    surface = CentralPatchSurface(**kwargs)

    if expected_shape is None:
        assert surface.shape is None
    elif issubclass(expected_shape, Shape):
        assert isinstance(surface.shape, expected_shape)
        # Shape BSDF is referenced
        assert isinstance(surface.shape.bsdf, Ref)
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
            match="A 'CentralPatchSurface' cannot be traversed if its 'shape' field is unset",
        ):
            traverse(surface)
    else:
        raise NotImplementedError


def test_central_patch_texture_scale(mode_mono):
    # Default value: the patch is not scaled
    surface = CentralPatchSurface(shape={"type": "rectangle", "edges": 10.0})
    assert np.allclose(1.0, surface._texture_scale())

    # Specify edge values
    surface = CentralPatchSurface(
        shape={"type": "rectangle", "edges": 10.0},
        patch_edges=[10.0, 10.0 / 3.0],
    )
    assert np.allclose(surface._texture_scale(), [1.0 / 3.0, 1.0])


def test_central_patch_scale_kernel_dict(mode_mono):
    surface = CentralPatchSurface(
        shape={"type": "rectangle", "edges": 3000.0 * ureg.km},
        patch_edges=100 * ureg.km,
        id="surface",
    )

    template, params = traverse(surface)
    kernel_dict = template.render(ctx=KernelContext())
    result = kernel_dict["surface_bsdf"]["weight"]["to_uv"].matrix
    expected = (
        mi.ScalarTransform4f.scale([10, 10, 1])
        @ mi.ScalarTransform4f.translate((-0.45, -0.45, 0))
    ).matrix

    assert dr.allclose(result, expected)
