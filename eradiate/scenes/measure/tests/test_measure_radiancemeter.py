from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._radiancemeter import RadiancemeterMeasure


def test_radiancemeter(mode_mono):
    # Instantiation succeeds
    s = RadiancemeterMeasure()

    # The kernel dict can be instantiated
    ctx = KernelDictContext()
    assert KernelDict.from_elements(s, ctx=ctx).load()
