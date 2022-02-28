import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import PerspectiveCameraMeasure


def test_perspective(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load()

    # Origin and target cannot be the same
    for point in [[0, 0, 0], [1, 1, 1], [-1, 0.5, 1.3333]]:
        with pytest.raises(ValueError):
            PerspectiveCameraMeasure(origin=point, target=point)

    # Up must differ from the camera's viewing direction
    with pytest.raises(ValueError):
        PerspectiveCameraMeasure(origin=[0, 1, 0], target=[1, 0, 0], up=[1, -1, 0])


def test_perspective_external_medium(mode_mono):
    # create a series of perspective camera measures and a kernel dict context
    # which places some of them inside and outside the atmospheric volume
    # assert that the external medium is set correctly in the cameras' kernel dicts

    s1 = PerspectiveCameraMeasure()

    ctx1 = KernelDictContext(atmosphere_medium_id="test_atmosphere")
    ctx2 = KernelDictContext()

    kd1 = s1.kernel_dict(ctx=ctx1)
    kd2 = s1.kernel_dict(ctx=ctx2)

    assert kd1["measure"]["medium"] == {"type": "ref", "id": "test_atmosphere"}
    assert "medium" not in kd2["measure"].keys()
