import mitsuba as mi

from eradiate import unit_registry as ureg
from eradiate.scenes.illumination import ConstantIllumination
from eradiate.scenes.spectra import UniformSpectrum
from eradiate.test_tools.types import check_scene_element


def test_constant_construct(modes_all):
    # Construction without argument succeeds
    illumination = ConstantIllumination()
    assert illumination
    assert isinstance(illumination.radiance, UniformSpectrum)
    assert illumination.radiance.value == 1.0 * ureg("W/m^2/sr/nm")


def test_constant_kernel_dict(modes_all_double):
    # The associated kernel dict is correctly formed and can be loaded
    illumination = ConstantIllumination()
    check_scene_element(illumination, mi.Emitter)
