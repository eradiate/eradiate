import mitsuba as mi

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import NodeSceneElement, traverse
from eradiate.scenes.illumination import ConstantIllumination, Illumination
from eradiate.scenes.spectra import UniformSpectrum
from eradiate.test_tools.types import check_type


def test_constant_type():
    check_type(
        ConstantIllumination,
        expected_mro=[Illumination, NodeSceneElement],
        expected_slots=[],
    )


def test_constant_construct(modes_all):
    # Construction without argument succeeds
    c = ConstantIllumination()
    assert c
    assert isinstance(c.radiance, UniformSpectrum)
    assert c.radiance.value == 1.0 * ureg("W/m^2/sr/nm")


def test_constant_kernel_dict(modes_all_double):
    # The associated kernel dict is correctly formed and can be loaded
    c = ConstantIllumination()
    template, _ = traverse(c)
    ctx = KernelDictContext()
    kernel_dict = template.render(ctx)
    assert isinstance(mi.load_dict(kernel_dict), mi.Emitter)
