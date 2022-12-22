import numpy as np
import pint
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.core import (
    BoundingBox,
    Param,
    ParameterMap,
    ParamFlags,
    render_params,
)


def test_render_params():
    param_map = {
        "foo": 0,
        "bar": Param(lambda ctx: ctx, ParamFlags.GEOMETRIC),
        "baz": Param(lambda ctx: ctx, ParamFlags.SPECTRAL),
    }

    # If no flags are passed, all params are rendered
    d = param_map.copy()
    unused = render_params(d, ctx=1, flags=ParamFlags.ALL)
    assert not unused
    assert d["bar"] == 1 and d["baz"] == 1

    # If a flag is passed, only the corresponding params are rendered
    d = param_map.copy()
    unused = render_params(d, ctx=1, flags=ParamFlags.SPECTRAL)
    assert unused == ["bar"]
    assert d["baz"] == 1
    assert "bar" in d

    # If drop is set to True, unused parameters are dropped
    d = param_map.copy()
    unused = render_params(d, ctx=1, flags=ParamFlags.SPECTRAL, drop=True)
    assert unused == ["bar"]
    assert d["baz"] == 1
    assert "bar" not in d


def test_parameter_map():
    parameter_map = ParameterMap(
        {
            "foo": 0,
            "foo.bar": 1,
            "bar": Param(lambda ctx: ctx, ParamFlags.GEOMETRIC),
            "baz": Param(lambda ctx: ctx, ParamFlags.SPECTRAL),
        }
    )

    # We can remove or keep selected parameters
    pmap = parameter_map.copy()
    pmap.remove(r"foo.*")
    assert pmap.keys() == {"bar", "baz"}

    pmap = parameter_map.copy()
    pmap.keep(r"foo.*")
    assert pmap.keys() == {"foo", "foo.bar"}


def test_bbox():
    # Instantiation with correctly ordered unitless values works
    bbox = BoundingBox([0, 0, 0], [1, 1, 1])
    assert bbox.units == ureg.m

    # Instantiation with correctly ordered unit-attached values works
    bbox = BoundingBox([0, 0, 0] * ureg.m, [1, 1, 1] * ureg.m)
    assert bbox.min.units == ureg.m

    # Extents are correctly computed
    assert np.allclose([1, 2, 3] * ureg.m, BoundingBox([0, 0, 0], [1, 2, 3]).extents)

    # Extent shapes must be compatible
    with pytest.raises(ValueError):
        BoundingBox([0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        BoundingBox([0, 0, 0], [[1, 1, 1]])

    # Unit mismatch raises
    with pytest.raises(pint.DimensionalityError):
        BoundingBox([0, 0] * ureg.m, [1, 1] * ureg.s)


def test_bbox_convert():
    bbox_ref = BoundingBox([0, 0, 0], [1, 1, 1])

    bbox_convert = BoundingBox.convert([[0, 0, 0], [1, 1, 1]])
    assert np.allclose(bbox_convert.min, bbox_ref.min)
    assert np.allclose(bbox_convert.max, bbox_ref.max)

    bbox_convert = BoundingBox.convert(np.array([[0, 0, 0], [1, 1, 1]]))
    assert np.allclose(bbox_convert.min, bbox_ref.min)
    assert np.allclose(bbox_convert.max, bbox_ref.max)

    bbox_ref = BoundingBox([0, 0, 0] * ureg.m, [1, 1, 1] * ureg.m)
    bbox_convert = BoundingBox.convert([[0, 0, 0], [1, 1, 1]] * ureg.m)
    assert np.allclose(bbox_convert.min, bbox_ref.min)
    assert np.allclose(bbox_convert.max, bbox_ref.max)


def test_bbox_contains():
    bbox = BoundingBox([0, 0, 0], [1, 1, 1])

    # Works with a single point
    assert bbox.contains([0.5, 0.5, 0.5])
    assert not bbox.contains([0.5, 0.5, -0.5])
    assert not bbox.contains([0.5, 0.5, 0.5] * ureg.km)

    # Works with multiple points
    assert np.all(bbox.contains([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]) == [True, False])
