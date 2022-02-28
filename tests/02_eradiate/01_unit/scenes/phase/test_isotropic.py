from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.phase import IsotropicPhaseFunction


def test_isotropic(modes_all):
    ctx = KernelDictContext()

    # Default constructor
    phase = IsotropicPhaseFunction()

    # Check if produced kernel dict can be instantiated
    kernel_dict = KernelDict.from_elements(phase, ctx=ctx)
    assert kernel_dict.load() is not None
