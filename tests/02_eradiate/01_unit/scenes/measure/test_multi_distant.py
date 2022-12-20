import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.frame import AzimuthConvention
from eradiate.scenes.core import NodeSceneElement
from eradiate.scenes.measure import measure_factory
from eradiate.scenes.measure._distant import DistantMeasure
from eradiate.scenes.measure._multi_distant import (
    AngleLayout,
    AzimuthRingLayout,
    DirectionLayout,
    GridLayout,
    HemispherePlaneLayout,
    Layout,
    MultiDistantMeasure,
)
from eradiate.test_tools.types import check_node_scene_element, check_type

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

    # Regular construction pattern succeeds
    layout = AngleLayout([[0, 0], [45, 0], [45, 90], [45, 180]] * ureg.deg)

    # Directions are correctly computed and point outwards
    assert np.allclose(
        layout.directions,
        [
            [0, 0, 1],
            [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        ],
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
    assert np.allclose(layout.angles, [[0, 0], [45, 0], [45, 90], [45, 180]] * ureg.deg)


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("east_right", [[1, 0, 0], [0, 1, 0]]),
        ("north_left", [[0, 1, 0], [1, 0, 0]]),
    ],
)
def test_layout_azimuth_convention(mode_mono, convention, expected):
    # Azimuth conventions are properly applied
    layout = AngleLayout(
        angles=[[90, 0], [90, 90]] * ureg.deg, azimuth_convention=convention
    )
    assert np.allclose(layout.directions, expected)


def test_direction_layout(mode_mono):
    """
    Unit tests for the DirectionLayout class
    """
    # Constructing without argument fails
    with pytest.raises(TypeError):
        print(DirectionLayout())

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
    assert np.allclose(
        layout.angles,
        [[0, 0], [45, 0], [45, 90], [45, 180]] * ureg.deg,
    )


def test_azimuth_ring_layout(mode_mono):
    # Constructing without arguments fails
    with pytest.raises(TypeError):
        AzimuthRingLayout()

    # Regular construction pattern succeeds
    layout = AzimuthRingLayout(
        45 * ureg.deg, np.linspace(0, 2 * np.pi, 8, endpoint=False) * ureg.rad
    )

    # Angles are correctly computed
    assert np.allclose(
        layout.angles,
        [
            [45, 0],
            [45, 45],
            [45, 90],
            [45, 135],
            [45, 180],
            [45, 225],
            [45, 270],
            [45, 315],
        ]
        * ureg.deg,
    )

    # We don't test the direction generation (it's the default implementation)


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


# ------------------------------------------------------------------------------
#                       MultiDistantMeasure implementation
# ------------------------------------------------------------------------------


def test_multi_distant_measure_type():
    check_type(
        MultiDistantMeasure,
        expected_mro=[DistantMeasure, NodeSceneElement],
        expected_slots=[],
    )


def test_multi_distant_measure_construct(mode_mono):
    """
    Basic constructor testing for MultiDistantMeasure.
    """
    # Constructing without argument succeeds
    measure = MultiDistantMeasure(direction_layout=[0, 45] * ureg.deg)
    check_node_scene_element(measure, mi.Sensor)


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


def test_multi_distant_measure_from_viewing_angles(mode_mono):
    """
    Unit tests for :meth:`.MultiDistantMeasure.from_viewing_angles`.
    """
    # Construct from viewing angles not in a hemisphere plane cut
    zeniths = [0, 45, 90, 45, 45, 45, 90, 90, 90]
    azimuths = [0, 0, 0, 0, 45, 90, 0, 45, 90]
    angles = np.reshape(np.stack((zeniths, azimuths), axis=-1), (-1, 1, 2)) * ureg.deg

    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert np.allclose(measure.viewing_angles, angles)

    # Specifying the hplane param will have the validation step raise
    with pytest.raises(TypeError):
        MultiDistantMeasure.from_viewing_angles(zeniths, azimuths, hplane=0.0)

    # Construct from viewing angles within the same plane using a single azimuth value
    zeniths = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60])
    azimuths = 0
    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert isinstance(measure.direction_layout, HemispherePlaneLayout)
    assert measure.direction_layout.azimuth == 0.0 * ureg.deg

    angles = (
        np.reshape(
            np.stack((zeniths, np.zeros_like(zeniths)), axis=-1),
            (-1, 1, 2),
        )
        * ureg.deg
    )
    assert np.allclose(measure.viewing_angles, angles)

    # Construct from viewing angles within the same plane using a single azimuth value
    zeniths = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60])
    azimuths = 180
    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert isinstance(measure.direction_layout, HemispherePlaneLayout)
    assert measure.direction_layout.azimuth == 180.0 * ureg.deg

    angles = (
        np.reshape(
            np.stack((zeniths, np.full_like(zeniths, 180)), axis=-1),
            (-1, 1, 2),
        )
        * ureg.deg
    )
    assert np.allclose(measure.viewing_angles, angles)

    # Construct an azimuthal ring
    zeniths = 45
    azimuths = np.arange(0, 360, 45)
    measure = MultiDistantMeasure.from_viewing_angles(zeniths, azimuths)
    assert isinstance(measure.direction_layout, AzimuthRingLayout)
    assert measure.direction_layout.zenith == 45 * ureg.deg

    angles = (
        np.reshape(
            np.stack((np.full_like(azimuths, zeniths), azimuths), axis=-1),
            (-1, 1, 2),
        )
        * ureg.deg
    )
    assert np.allclose(measure.viewing_angles, angles)


def test_multi_distant_measure_from_viewing_angles_convention(mode_mono):
    # Construct from viewing angles within the same plane using a single azimuth value
    measure = MultiDistantMeasure.from_viewing_angles(
        zeniths=[-60, -45, -30, -15, 0, 15, 30, 45, 60] * ureg.deg,
        azimuths=0.0,
        azimuth_convention="north_left",
    )

    # The requested azimuth convention is correctly passed to the constructor
    assert measure.direction_layout.azimuth_convention is AzimuthConvention.NORTH_LEFT

    # Selecting a non-standard convention doesn't modify the hemispherical plane
    # azimuth and viewing angles are not modified (beyond the hemispherical
    # plane transform)
    assert isinstance(measure.direction_layout, HemispherePlaneLayout)
    assert measure.direction_layout.azimuth == 0.0 * ureg.deg
    expected = [
        [-60, 0],
        [-45, 0],
        [-30, 0],
        [-15, 0],
        [0, 0],
        [15, 0],
        [30, 0],
        [45, 0],
        [60, 0],
    ] * ureg.deg
    result = measure.viewing_angles.squeeze()
    assert np.allclose(result, expected)

    # Another check of the azimuth values: computed viewing angles are expressed
    # in the specified convention
    measure = MultiDistantMeasure.from_viewing_angles(
        zeniths=45.0 * ureg.deg,
        azimuths=[0, 45, 90, 180, 360] * ureg.deg,
        azimuth_convention="north_left",
    )
    expected = [[45, 0], [45, 45], [45, 90], [45, 180], [45, 0]] * ureg.deg
    result = measure.viewing_angles.squeeze()
    assert np.allclose(result, expected)

    # Check that generated directions are correct: the constructor internally
    # performs a transform to East right
    measure = MultiDistantMeasure.from_viewing_angles(
        zeniths=90.0 * ureg.deg,
        azimuths=[0, 45, 90, 180, 360] * ureg.deg,
        azimuth_convention="north_left",
    )
    assert np.allclose(
        measure.direction_layout.directions,
        [
            [0, 1, 0],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
        ],
    )
