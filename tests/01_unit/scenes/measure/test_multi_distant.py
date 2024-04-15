import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.measure import (
    AngleLayout,
    AzimuthRingLayout,
    DirectionLayout,
    GridLayout,
    HemispherePlaneLayout,
    Layout,
    MultiDistantMeasure,
    measure_factory,
)
from eradiate.test_tools.types import check_scene_element

# ------------------------------------------------------------------------------
#                               Layout framework
# ------------------------------------------------------------------------------


def test_angle_layout(mode_mono):
    """
    Unit tests for the AngleLayout class
    """
    # Constructing without argument fails
    with pytest.raises(TypeError):
        AngleLayout()

    # (2,) arrays are reshaped as (1, 2)
    assert AngleLayout([0, 0] * ureg.deg).angles.shape == (1, 2)

    # Zenith values outside [0, 180]° are not allowed
    with pytest.raises(ValueError):
        AngleLayout([-45, 0] * ureg.deg)
    with pytest.raises(ValueError):
        AngleLayout([210, 0] * ureg.deg)

    # Regular construction pattern succeeds
    layout = AngleLayout([[0, 0], [45, 0], [45, 90], [45, 180]] * ureg.deg)

    # Directions are correctly computed and point outwards
    np.testing.assert_allclose(
        layout.directions,
        [
            [0, 0, 1],
            [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        ],
        atol=1e-8,
    )

    # Dictionary conversion protocol applies units
    layout = Layout.convert(
        {
            "type": "angles",
            "angles": [
                [0, 0],
                [np.pi / 4, 0],
                [np.pi / 4, np.pi / 2],
                [np.pi / 4, np.pi],
            ],
            "angles_units": "rad",
        }
    )
    np.testing.assert_allclose(
        layout.angles.m_as(ureg.deg),
        [[0, 0], [45, 0], [45, 90], [45, 180]],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("east_right", [[1, 0, 0], [0, 1, 0]]),
        ("north_left", [[0, 1, 0], [1, 0, 0]]),
    ],
)
def test_angle_layout_azimuth_convention(mode_mono, convention, expected):
    # Azimuth conventions are properly applied
    layout = AngleLayout(
        angles=[[90, 0], [90, 90]] * ureg.deg, azimuth_convention=convention
    )
    np.testing.assert_allclose(layout.directions, expected, atol=1e-8)


def test_direction_layout(mode_mono):
    """
    Unit tests for the DirectionLayout class
    """
    # Constructing without argument fails
    with pytest.raises(TypeError):
        DirectionLayout()

    # (3,) arrays are reshaped as (1, 3)
    assert DirectionLayout([0, 0, 1]).directions.shape == (1, 3)

    # Regular construction pattern succeeds
    layout = DirectionLayout(
        [
            [0, 0, 1],
            [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        ],
    )

    # Angles are correctly computed
    np.testing.assert_allclose(
        layout.angles.m_as(ureg.deg),
        [[0, 0], [45, 0], [45, 90], [45, 180]],
    )


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("east_right", [[90, 0], [90, 90]]),
        ("north_left", [[90, 90], [90, 0]]),
    ],
)
def test_direction_layout_azimuth_convention(mode_mono, convention, expected):
    # Azimuth conventions are properly applied
    layout = DirectionLayout(
        directions=[[1, 0, 0], [0, 1, 0]], azimuth_convention=convention
    )
    np.testing.assert_allclose(layout.angles.m_as(ureg.deg), expected, atol=1e-8)


def test_azimuth_ring_layout(mode_mono):
    # Constructing without arguments fails
    with pytest.raises(TypeError):
        AzimuthRingLayout()

    # Regular construction pattern succeeds
    layout = AzimuthRingLayout(
        45 * ureg.deg, np.linspace(0, 2 * np.pi, 8, endpoint=False) * ureg.rad
    )

    # Angles are correctly computed
    np.testing.assert_allclose(
        layout.angles.m_as(ureg.deg),
        [
            [45, 0],
            [45, 45],
            [45, 90],
            [45, 135],
            [45, 180],
            [45, 225],
            [45, 270],
            [45, 315],
        ],
    )

    # We don't test the direction generation (it's the default implementation)


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("east_right", [[1, 0, 0], [0, 1, 0]]),
        ("north_left", [[0, 1, 0], [1, 0, 0]]),
    ],
)
def test_azimuth_ring_layout_azimuth_convention(mode_mono, convention, expected):
    # Azimuth conventions are properly applied
    layout = AzimuthRingLayout(
        90 * ureg.deg,
        [0, 90] * ureg.deg,
        azimuth_convention=convention,
    )
    np.testing.assert_allclose(layout.directions, expected, atol=1e-8)


def test_hemisphere_plane_layout(mode_mono):
    # Constructing without arguments fails
    with pytest.raises(TypeError):
        HemispherePlaneLayout()

    # Regular construction pattern succeeds
    layout = HemispherePlaneLayout(
        np.linspace(-90, 90, 5, endpoint=True) * ureg.deg, 0 * ureg.deg
    )

    # Angles are correctly computed
    assert np.allclose(
        layout.angles,
        [[-90, 0], [-45, 0], [0, 0], [45, 0], [90, 0]] * ureg.deg,
    )

    # We don't test the direction generation (it's the default implementation)


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("east_right", [[-1, 0, 0], [0, 0, 1], [1, 0, 0]]),
        ("north_left", [[0, -1, 0], [0, 0, 1], [0, 1, 0]]),
    ],
)
def test_hemisphere_plane_layout_azimuth_convention(mode_mono, convention, expected):
    # Azimuth conventions are properly applied
    layout = HemispherePlaneLayout(
        [-90, 0, 90] * ureg.deg,
        0 * ureg.deg,
        azimuth_convention=convention,
    )
    np.testing.assert_allclose(layout.directions, expected, atol=1e-8)


def test_grid_layout(mode_mono):
    # Constructing without arguments fails
    with pytest.raises(TypeError):
        GridLayout()

    # Regular construction pattern succeeds
    layout = GridLayout([0, 45] * ureg.deg, [0, 180] * ureg.deg)

    # Angles are correctly computed
    assert np.allclose(
        layout.angles,
        [[0, 0], [45, 0], [0, 180], [45, 180]] * ureg.deg,
    )

    # We don't test the direction generation (it's the default implementation)


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("east_right", [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        ("north_left", [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    ],
)
def test_grid_layout_azimuth_convention(mode_mono, convention, expected):
    # Azimuth conventions are properly applied
    layout = GridLayout(
        [0, 90] * ureg.deg,
        [0, 90] * ureg.deg,
        azimuth_convention=convention,
    )
    np.testing.assert_allclose(layout.directions, expected, atol=1e-8)


# ------------------------------------------------------------------------------
#                       MultiDistantMeasure implementation
# ------------------------------------------------------------------------------


def test_multi_distant_measure_construct(mode_mono):
    """
    Basic constructor testing for MultiDistantMeasure.
    """
    # Constructing without argument succeeds
    measure = MultiDistantMeasure(direction_layout=[0, 45] * ureg.deg)
    check_scene_element(measure, mi.Sensor)


def test_multi_distant_measure_construct_specific(mode_mono):
    """
    Class method constructor testing for MultiDistantMeasure
    """
    # Unitless values are attached default units
    measure = MultiDistantMeasure.hplane(zeniths=[0, 45], azimuth=0)
    assert isinstance(measure.direction_layout, HemispherePlaneLayout)
    assert np.allclose(measure.direction_layout.angles, [[0, 0], [45, 0]] * ureg.deg)

    # The dictionary conversion protocol applies units, including when used with
    # a class method constructor
    measure = measure_factory.convert(
        {
            "type": "mdistant",
            "construct": "from_angles",
            "angles": [np.pi / 4, 0],
            "angles_units": "rad",
        }
    )
    assert isinstance(measure.direction_layout, AngleLayout)
    assert np.allclose(measure.direction_layout.angles, [[45, 0]] * ureg.deg)


def test_multi_distant_measure_viewing_angles(mode_mono):
    """
    Unit tests for :attr:`.MultiDistantMeasure.viewing_angles`.
    """
    # Viewing angle computation is correct
    measure = MultiDistantMeasure(
        direction_layout=[
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    expected = (
        np.array(
            [
                [0, 0],
                [45, 0],
                [45, 90],
                [np.rad2deg(np.arccos(1 / np.sqrt(3))), 45],
            ]
        ).reshape((4, 1, 2))
        * ureg.deg
    )

    assert np.allclose(measure.viewing_angles, expected)

    # Directions which would normally map to the [-π, 0] domain are normalised
    # to [0, 2π]
    measure = MultiDistantMeasure(direction_layout=[[0, -1, 1]])
    expected = [45, 270] * ureg.deg
    assert np.allclose(measure.viewing_angles, expected)
