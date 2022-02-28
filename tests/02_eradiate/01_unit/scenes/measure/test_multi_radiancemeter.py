from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._multi_radiancemeter import MultiRadiancemeterMeasure


def test_multi_radiancemeter_noargs(mode_mono):
    # Instantiation succeeds
    s = MultiRadiancemeterMeasure()

    # The kernel dict can be instantiated
    ctx = KernelDictContext()
    assert KernelDict.from_elements(s, ctx=ctx).load()


def test_multi_radiancemeter_args(mode_mono):
    # Instantiation succeeds
    s = MultiRadiancemeterMeasure(
        origins=[[0, 0, 0]] * 3, directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    # The kernel dict can be instantiated
    ctx = KernelDictContext()
    assert KernelDict.from_elements(s, ctx=ctx).load()


def test_multi_radiancemeter_external_medium(mode_mono):
    # create a series of multiradiancemeter measures and a kernel dict context
    # which places some of them inside and outside the atmospheric volume
    # assert that the external medium is set correctly in the cameras' kernel dicts

    d1 = MultiRadiancemeterMeasure()

    ctx1 = KernelDictContext(atmosphere_medium_id="test_atmosphere")
    ctx2 = KernelDictContext()

    kd1 = d1.kernel_dict(ctx=ctx1)
    kd2 = d1.kernel_dict(ctx=ctx2)

    assert kd1["measure"]["medium"] == {"type": "ref", "id": "test_atmosphere"}
    assert "medium" not in kd2["measure"].keys()
