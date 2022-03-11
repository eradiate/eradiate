import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.phase import RayleighPhaseFunction


def test_rayleigh(modes_all):
    ctx = KernelDictContext()

    # Default constructor
    phase = RayleighPhaseFunction()

    # Check if produced kernel dict can be instantiated
    kernel_dict = KernelDict.from_elements(phase, ctx=ctx)
    assert isinstance(kernel_dict.load(), mi.PhaseFunction)
