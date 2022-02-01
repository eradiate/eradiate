from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._radiancemeter import RadiancemeterMeasure


def test_radiancemeter(mode_mono):
    # Instantiation succeeds
    s = RadiancemeterMeasure()

    # The kernel dict can be instantiated
    ctx = KernelDictContext()
    assert KernelDict.from_elements(s, ctx=ctx).load()


def test_radiancemeter_external_medium(mode_mono):
    # create a series of radiancemeter measures and a kernel dict context
    # which places some of them inside and outside the atmospheric volume
    # assert that the external medium is set correctly in the cameras' kernel dicts

    s1 = RadiancemeterMeasure()

    ctx1 = KernelDictContext(atmosphere_medium_id="test_atmosphere")
    ctx2 = KernelDictContext()

    kd1 = s1.kernel_dict(ctx=ctx1)
    kd2 = s1.kernel_dict(ctx=ctx2)

    assert kd1["measure"]["medium"] == {"type": "ref", "id": "test_atmosphere"}
    assert "medium" not in kd2["measure"].keys()
