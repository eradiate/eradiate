from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.phase import RayleighPhaseFunction


def test_phase_rayleigh(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    phase = RayleighPhaseFunction()

    # Check if produced kernel dict can be instantiated
    kernel_dict = KernelDict.new(phase, ctx=ctx)
    assert kernel_dict.load() is not None
