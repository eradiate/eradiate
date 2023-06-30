import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.illumination import AstroObjectIllumination
from eradiate.scenes.spectra import SolarIrradianceSpectrum, UniformSpectrum
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs, expected_irradiance_type, angular_diameter",
    [({}, SolarIrradianceSpectrum, 1.0), ({"irradiance": 1.0}, UniformSpectrum, 2.0)],
    ids=["noargs", "from_scalar"],
)
def test_astro_object_construct(
    modes_all, kwargs, expected_irradiance_type, angular_diameter
):
    # Construction without argument succeeds
    illumination = AstroObjectIllumination(**kwargs, angular_diameter=angular_diameter)
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
def test_astro_object_to_world(mode_mono, zenith, azimuth, direction, to_world):
    """
    Direction is correctly converted to transformation matrix.
    """
    illumination = AstroObjectIllumination(zenith=zenith, azimuth=azimuth)
    np.testing.assert_allclose(illumination.direction, direction)
    np.testing.assert_allclose(illumination._to_world.matrix, to_world)


def test_astro_object_kernel_dict(modes_all_double):
    # The associated kernel dict is correctly formed and can be loaded
    illumination = AstroObjectIllumination()
    check_scene_element(illumination, mi_cls=mi.Emitter)


COS_PI_4 = 0.5 * np.sqrt(2)


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
)
def test_astro_object_azimuth_convention(mode_mono, azimuth_convention, expected):
    illumination = AstroObjectIllumination(
        zenith=45 * ureg.deg,
        azimuth=90 * ureg.deg,
        azimuth_convention=azimuth_convention,
    )
    assert np.allclose(illumination.direction, expected), illumination.direction
