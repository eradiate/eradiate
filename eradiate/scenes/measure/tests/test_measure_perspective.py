import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import PerspectiveCameraMeasure


def test_perspective(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    ctx = KernelDictContext()
    assert KernelDict.new(d, ctx=ctx).load() is not None

    # Origin and target cannot be the same
    for point in [[0, 0, 0], [1, 1, 1], [-1, 0.5, 1.3333]]:
        with pytest.raises(ValueError):
            PerspectiveCameraMeasure(origin=point, target=point)

    # Up must differ from the camera's viewing direction
    with pytest.raises(ValueError):
        PerspectiveCameraMeasure(origin=[0, 1, 0], target=[1, 0, 0], up=[1, -1, 0])
