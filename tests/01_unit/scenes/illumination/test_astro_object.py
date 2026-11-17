import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.illumination import AstroObjectIllumination
from eradiate.scenes.spectra import SolarIrradianceSpectrum, UniformSpectrum
from eradiate.testing.types import check_scene_element

COS_PI_4 = 0.5 * np.sqrt(2)


class TestAstroObject:
    @pytest.mark.parametrize(
        "kwargs, expected_irradiance_type, angular_diameter",
        [
            ({}, SolarIrradianceSpectrum, 1.0),
            ({"irradiance": 1.0}, UniformSpectrum, 2.0),
        ],
        ids=["noargs", "from_scalar"],
    )
    def test_construct(
        self, modes_all, kwargs, expected_irradiance_type, angular_diameter
    ):
        # Construction without argument succeeds
        illumination = AstroObjectIllumination(
            **kwargs, angular_diameter=angular_diameter
        )
        assert illumination
        assert isinstance(illumination.irradiance, expected_irradiance_type)
        assert illumination.angular_diameter == angular_diameter * ureg.deg

    @pytest.mark.parametrize(
        "zenith, azimuth, direction, to_world",
        [
            (
                0.0 * ureg.deg,
                0.0 * ureg.deg,
                [0, 0, 1],
                [
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            ),
            (
                30 * ureg.deg,
                0.0 * ureg.deg,
                [0.5, 0.0, np.sqrt(3) / 2],
                [
                    [0.0, np.sqrt(3) / 2, 0.5, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -0.5, np.sqrt(3) / 2, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
        ],
    )
    def test_to_world(self, mode_mono, zenith, azimuth, direction, to_world):
        """
        Direction is correctly converted to transformation matrix.
        """
        illumination = AstroObjectIllumination(zenith=zenith, azimuth=azimuth)
        np.testing.assert_allclose(illumination.direction, direction)
        np.testing.assert_allclose(illumination._to_world.matrix, to_world)

    @pytest.mark.parametrize("sampling_weight", [None, 0.5])
    def test_kernel_dict(self, modes_all_double, sampling_weight):
        # The associated kernel dict is correctly formed and can be loaded
        illumination = AstroObjectIllumination(sampling_weight=sampling_weight)
        mi_obj = check_scene_element(illumination, mi_cls=mi.Emitter)

        # Check sampling weight correctness
        expected = 1.0 if sampling_weight is None else sampling_weight
        assert mi_obj.obj.sampling_weight() == expected

    @pytest.mark.parametrize(
        "azimuth_convention, expected",
        [
            ("east_right", [0, COS_PI_4, COS_PI_4]),
            ("east_left", [0, -COS_PI_4, COS_PI_4]),
            ("north_right", [-COS_PI_4, 0, COS_PI_4]),
            ("north_left", [COS_PI_4, 0, COS_PI_4]),
            ("west_right", [0, -COS_PI_4, COS_PI_4]),
            ("west_left", [0, COS_PI_4, COS_PI_4]),
            ("south_right", [COS_PI_4, 0, COS_PI_4]),
            ("south_left", [-COS_PI_4, 0, COS_PI_4]),
        ],
        ids=[
            "east_right",
            "east_left",
            "north_right",
            "north_left",
            "west_right",
            "west_left",
            "south_right",
            "south_left",
        ],
    )
    def test_azimuth_convention(self, mode_mono, azimuth_convention, expected):
        illumination = AstroObjectIllumination(
            zenith=45 * ureg.deg,
            azimuth=90 * ureg.deg,
            azimuth_convention=azimuth_convention,
        )
        assert np.allclose(illumination.direction, expected), illumination.direction

    @pytest.mark.parametrize(
        "azimuth",
        [-90.0 * ureg.deg, -np.pi / 2 * ureg.rad],
        ids=["-90 deg", "-pi/2 rad"],
    )
    def test_azimuth_negative_values(self, mode_mono, azimuth):
        with pytest.warns(UserWarning):
            illumination = AstroObjectIllumination(
                azimuth=ureg.Quantity(azimuth), zenith=0.0 * ureg.deg
            )
            np.testing.assert_allclose(illumination.azimuth.m_as("deg"), 270.0)
