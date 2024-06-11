import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.measure import DistantMeasure
from eradiate.test_tools.types import check_scene_element


def test_distant_construct(mode_mono):
    # Constructor without arguments succeeds
    measure = DistantMeasure()
    check_scene_element(measure, mi.Sensor)

    # The from_angles() constructor sets the direction as expected
    measure = DistantMeasure.from_angles((45.0, 90.0))
    d = np.array([0.0, 1.0, 1.0])
    d /= np.linalg.norm(d)
    np.testing.assert_allclose(measure.direction, d, atol=1e-9)
    check_scene_element(measure, mi.Sensor)

    # It also applies the azimuth convention
    measure = DistantMeasure.from_angles(
        angles=(45.0, 90.0), azimuth_convention="north_left"
    )
    d = np.array([1.0, 0.0, 1.0])
    d /= np.linalg.norm(d)
    np.testing.assert_allclose(measure.direction, d, atol=1e-9)
    check_scene_element(measure, mi.Sensor)


@pytest.mark.parametrize(
    "direction, azimuth_convention, expected",
    [
        ([0, 0, 1], "east_right", [[[0, 0]]]),
        ([0, 1, 1], "east_right", [[[45, 90]]]),
        ([0, 1, 1], "north_left", [[[45, 0]]]),
    ],
)
def test_distant_angles(mode_mono, direction, azimuth_convention, expected):
    measure = DistantMeasure(direction=direction, azimuth_convention=azimuth_convention)
    np.testing.assert_allclose(measure.viewing_angles.m_as("deg"), expected)


def test_distant_full_scene(mode_mono):
    # This is a smoke test that runs a very simple simulation to check that the
    # full processing chain runs without crashing
    exp = AtmosphereExperiment(
        atmosphere=None,
        illumination={"type": "directional", "irradiance": np.pi},
        surface={"type": "lambertian", "reflectance": 1.0},
        measures={"type": "distant"},
    )
    result = eradiate.run(exp, spp=1)
    np.testing.assert_allclose(result.radiance.squeeze().values, 1.0)
