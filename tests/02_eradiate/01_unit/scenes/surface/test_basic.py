import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.surface import BasicSurface


def test_basic_surface_construct(modes_all_double):
    ctx = KernelDictContext()

    # Default constructor
    surface = BasicSurface()
    # The default value for `shape` is invalid: a shape must be manually
    # specified
    assert surface.shape is None

    with pytest.raises(ValueError):
        surface.kernel_dict(ctx).load()

    # Specify shape
    surface = BasicSurface(shape={"type": "rectangle"})
    kernel_dict = surface.kernel_dict(ctx)
    # The BSDF is referenced
    assert kernel_dict.data[surface.shape_id]["bsdf"]["type"] == "ref"
    # The constructed dict can be loaded
    assert kernel_dict.load()
