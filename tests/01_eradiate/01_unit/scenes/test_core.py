import numpy as np
import pint
import pytest

from eradiate import unit_registry as ureg
from eradiate._factory import Factory
from eradiate.scenes.core import (
    BoundingBox,
    get_factory,
)


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


def test_get_factory():
    """
    Check that all declared factories can be retrieved.
    """
    for element_type in [
        "atmosphere",
        "biosphere",
        "bsdf",
        "illumination",
        "integrator",
        "measure",
        "phase",
        "shape",
        "spectrum",
        "surface",
    ]:
        assert isinstance(get_factory(element_type), Factory)
