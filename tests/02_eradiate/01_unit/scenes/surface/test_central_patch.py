import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.surface import CentralPatchSurface
from eradiate.units import unit_registry as ureg


def test_central_patch_construct(modes_all_double):
    ctx = KernelDictContext()

    # Default constructor
    surface = CentralPatchSurface()
    # The default value for `shape` is invalid: a shape must be manually
    # specified
    assert surface.shape is None

    with pytest.raises(ValueError):
        surface.kernel_dict(ctx).load()

    # Specify shape
    surface = CentralPatchSurface(shape={"type": "rectangle"})
    kernel_dict = surface.kernel_dict(ctx)
    # The BSDF is referenced
    assert kernel_dict.data[surface.shape_id]["bsdf"]["type"] == "ref"
    # The constructed dict can be loaded
    assert kernel_dict.load()

    # Specify patch parameters
    # -- Square patch with edges specified with a scalar
    surface = CentralPatchSurface(shape={"type": "rectangle"}, patch_edges=1.0)
    assert surface.kernel_dict(ctx).load()
    # -- Rectangular patch with edges specified with a 2-vector
    surface = CentralPatchSurface(shape={"type": "rectangle"}, patch_edges=[1.0, 0.5])
    assert surface.kernel_dict(ctx).load()


def test_central_patch_texture_scale(mode_mono):
    ctx = KernelDictContext()

    # Default value: the patch is not scaled
    surface = CentralPatchSurface(shape={"type": "rectangle", "edges": 10.0})
    assert np.allclose(1.0, surface._texture_scale())

    # Specify edge values
    surface = CentralPatchSurface(
        shape={"type": "rectangle", "edges": 10.0},
        patch_edges=[10.0, 10.0 / 3.0],
    )
    assert np.allclose([1.0 / 3.0, 1.0], surface._texture_scale())


def test_central_patch_scale_kernel_dict(mode_mono):
    ctx = KernelDictContext()

    surface = CentralPatchSurface(
        shape={"type": "rectangle", "edges": 3000.0 * ureg.km},
        patch_edges=100 * ureg.km,
        id="surface",
    )

    kernel_dict = surface.kernel_bsdfs(ctx=ctx)
    result = kernel_dict["bsdf_surface"]["weight"]["to_uv"].matrix
    expected = (
        mi.ScalarTransform4f.scale([10, 10, 1])
        * mi.ScalarTransform4f.translate((-0.45, -0.45, 0))
    ).matrix

    assert dr.allclose(expected, result)
