import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.illumination import SpotIllumination
from eradiate.testing.types import check_scene_element


class TestSpot:
    @pytest.mark.parametrize("sampling_weight", [None, 0.5])
    def test_construct_basic(self, mode_mono, sampling_weight):
        # Construct without parameters
        illumination = SpotIllumination()
        check_scene_element(illumination, mi_cls=mi.Emitter)

        # Define all parameters
        illumination = SpotIllumination(
            target=[0, 0, 0],
            origin=[1, 1, 1],
            up=[1, 0, 1],
            beam_width=10 * ureg.deg,
            intensity=10,
            sampling_weight=sampling_weight,
        )
        mi_obj = check_scene_element(illumination, mi_cls=mi.Emitter)

        # Check sampling weight correctness
        expected = 1.0 if sampling_weight is None else sampling_weight
        assert mi_obj.obj.sampling_weight() == expected

    def test_construct_texture(self, mode_mono):
        # Construct with beam profile from file (resolved by file resolver)
        illumination = SpotIllumination(
            target=[0, 0, 0],
            origin=[1, 1, 1],
            up=[1, 0, 1],
            beam_width=10 * ureg.deg,
            intensity=10,
            beam_profile="texture/gaussian_3sigma.bmp",
        )
        check_scene_element(illumination, mi_cls=mi.Emitter)

    def test_construct_from_size(self, mode_mono):
        spot_radius = 1 * ureg.m
        beam_angle = 3 * ureg.deg

        # spot_radius / tan(beam_angle/2.)
        expected_distance = 38.18846

        illumination = SpotIllumination.from_size_at_target(
            target=[0, 0, 0],
            direction=[0, 0, -1],
            spot_radius=spot_radius,
            beam_width=beam_angle,
        )

        assert np.allclose(illumination.origin.m_as(ureg.m), [0, 0, expected_distance])

        check_scene_element(illumination, mi_cls=mi.Emitter)
