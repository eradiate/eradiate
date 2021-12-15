from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.phase import HenyeyGreensteinPhaseFunction


def test_hg(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    phase = HenyeyGreensteinPhaseFunction()

    # Construct with custom asymmetry parameter
    phase = HenyeyGreensteinPhaseFunction(g=0.25)

    # Check if produced kernel dict can be instantiated
    kernel_dict = KernelDict.from_elements(phase, ctx=ctx)
    assert kernel_dict.load() is not None
