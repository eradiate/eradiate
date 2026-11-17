import mitsuba as mi
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.illumination import ConstantIllumination
from eradiate.scenes.spectra import UniformSpectrum
from eradiate.testing.types import check_scene_element


class TestConstant:
    def test_construct(self, modes_all):
        # Construction without argument succeeds
        illumination = ConstantIllumination()
        assert illumination
        assert isinstance(illumination.radiance, UniformSpectrum)
        assert illumination.radiance.value == 1.0 * ureg("W/m^2/sr/nm")

    @pytest.mark.parametrize("sampling_weight", [None, 0.5])
    def test_kernel_dict(self, modes_all_double, sampling_weight):
        # The associated kernel dict is correctly formed and can be loaded
        illumination = ConstantIllumination(sampling_weight=sampling_weight)
        mi_obj = check_scene_element(illumination, mi.Emitter)

        # Check sampling weight correctness
        expected = 1.0 if sampling_weight is None else sampling_weight
        assert mi_obj.obj.sampling_weight() == expected
