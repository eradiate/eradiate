import mitsuba as mi

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import ConstantIllumination


def test_constant(mode_mono):
    # Constructor
    c = ConstantIllumination()
    ctx = KernelDictContext()
    assert c.kernel_dict(ctx)[c.id] == {
        "type": "constant",
        "radiance": {"type": "uniform", "value": 1.0},
    }
    assert isinstance(KernelDict.from_elements(c, ctx=ctx).load(), mi.Emitter)

    # Check if a more detailed spec is valid
    c = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert isinstance(KernelDict.from_elements(c, ctx=ctx).load(), mi.Emitter)

    # Check if 'uniform' shortcut works
    c = ConstantIllumination(radiance={"type": "uniform", "value": 1.0})
    assert isinstance(KernelDict.from_elements(c, ctx=ctx).load(), mi.Emitter)

    # Check if super lazy way works too
    c = ConstantIllumination(radiance=1.0)
    assert isinstance(KernelDict.from_elements(c, ctx=ctx).load(), mi.Emitter)
