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
