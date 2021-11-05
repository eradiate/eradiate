import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.frame import (
    angles_in_hplane,
    angles_to_direction,
    cos_angle_to_direction,
    direction_to_angles,
    spherical_to_cartesian,
)


def test_cos_angle_to_direction():
    # Old-style call
    assert np.allclose(cos_angle_to_direction(1.0, 0.0), [0, 0, 1])
    assert np.allclose(cos_angle_to_direction(0.0, 0.0), [1, 0, 0])
    assert np.allclose(cos_angle_to_direction(0.5, 0.0), [np.sqrt(3) / 2, 0, 0.5])
    assert np.allclose(
        cos_angle_to_direction(-1.0, ureg.Quantity(135.0, "deg")), [0, 0, -1]
    )

    # Vectorised call
    assert np.allclose(
        cos_angle_to_direction([1.0, 0.0, 0.5, -1.0], [0, 0, 0.0, 0.75 * np.pi]),
        ([0, 0, 1], [1, 0, 0], [np.sqrt(3) / 2, 0, 0.5], [0, 0, -1]),
    )


def test_angles_to_direction():
    # Old-style call
    assert np.allclose(angles_to_direction([0.0, 0.0]), [0, 0, 1])
    assert np.allclose(angles_to_direction([np.pi, 0.0]), [0, 0, -1])
    assert np.allclose(angles_to_direction([0.5 * np.pi, 0.0]), [1, 0, 0])
    assert np.allclose(angles_to_direction([0.5 * np.pi, np.pi]), [-1, 0, 0])
    assert np.allclose(angles_to_direction([0.5 * np.pi, 0.5 * np.pi]), [0, 1, 0])
    assert np.allclose(angles_to_direction([0.5 * np.pi, -0.5 * np.pi]), [0, -1, 0])
    assert np.allclose(
        angles_to_direction([0.25 * np.pi, 0]), [0.70710678, 0, 0.70710678]
    )
    assert np.allclose(
        angles_to_direction([0.5 * np.pi, 0.25 * np.pi]), [0.70710678, 0.70710678, 0]
    )

    # Vectorised call
    assert np.allclose(
        angles_to_direction(
            [
                [0.0, 0.0],
                [np.pi, 0.0],
                [0.5 * np.pi, 0.0],
                [0.5 * np.pi, np.pi],
                [0.5 * np.pi, 0.5 * np.pi],
                [0.5 * np.pi, -0.5 * np.pi],
                [0.25 * np.pi, 0],
                [0.5 * np.pi, 0.25 * np.pi],
                [-0.25 * np.pi, 0],
            ]
        ),
        [
            [0, 0, 1],
            [0, 0, -1],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [np.sqrt(2) * 0.5, 0, np.sqrt(2) * 0.5],
            [np.sqrt(2) * 0.5, np.sqrt(2) * 0.5, 0],
            [-np.sqrt(2) * 0.5, 0, np.sqrt(2) * 0.5],
        ],
    )


def test_direction_to_angles():
    # Old-style call
    assert np.allclose(direction_to_angles([0, 0, 1]), (0, 0))
    assert np.allclose(
        direction_to_angles([np.sqrt(2) / 2, np.sqrt(2) / 2, 0]).m_as(ureg.deg),
        [90.0, 45.0],
    )
    assert np.allclose(
        direction_to_angles([1, 1, 0]).m_as(ureg.deg),
        [90.0, 45.0],
    )
    assert np.allclose(
        direction_to_angles(
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        ).m_as(ureg.rad),
        [np.arccos(1 / np.sqrt(3)), 0.25 * np.pi],
    )
    assert np.allclose(
        direction_to_angles([1, 1, 1]).m_as(ureg.rad),
        [np.arccos(1 / np.sqrt(3)), 0.25 * np.pi],
    )

    # Vectorised call
    assert np.allclose(
        direction_to_angles(
            [
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [0, 0, -1],
            ]
        ),
        [
            [0, 0],
            [0.25 * np.pi, 0],
            [0.25 * np.pi, 0.5 * np.pi],
            [0.5 * np.pi, 0],
            [0.5 * np.pi, 0.5 * np.pi],
            [np.arccos(1 / np.sqrt(3)), 0.25 * np.pi],
            [np.pi, 0.0],
        ],
    )


def test_spherical_to_cartesian():
    r = 2.0
    theta = np.deg2rad(30)
    phi = np.deg2rad(0)
    d = spherical_to_cartesian(r, theta, phi)
    assert np.allclose(d, [1, 0, np.sqrt(3)])

    r = ureg.Quantity(2.0, "km")
    theta = np.deg2rad(60)
    phi = np.deg2rad(30)
    d = spherical_to_cartesian(r, theta, phi)
    assert np.allclose(d, ureg.Quantity([3.0 / 2.0, np.sqrt(3) / 2.0, 1.0], "km"))


@pytest.mark.parametrize(
    "theta, phi, plane, expected_plane",
    [
        (0, 0, 0, "p"),
        (45, 0, 0, "p"),
        (-45, 0, 0, "n"),
        (90, 0, 0, "p"),
        (-90, 0, 0, "n"),
        (0, 180, 0, "p"),
        (45, 180, 0, "n"),
        (-45, 180, 0, "p"),
        (90, 180, 0, "n"),
        (-90, 180, 0, "p"),
        (0, 0, 180, "p"),
        (45, 0, 180, "n"),
        (-45, 0, 180, "p"),
        (90, 0, 180, "n"),
        (-90, 0, 180, "p"),
        (0, 180, 180, "p"),
        (45, 180, 180, "p"),
        (-45, 180, 180, "n"),
        (90, 180, 180, "p"),
        (-90, 180, 180, "n"),
        (0, 0, 45, "p"),
        (45, 0, 45, None),
        (-45, 0, 45, None),
        (45, 0, -45, None),
        (-45, 0, -45, None),
    ],
)
def test_angles_in_hplane(theta, phi, plane, expected_plane):
    p, n = angles_in_hplane(plane, theta, phi, raise_exc=False)

    if expected_plane == "p":
        assert p and not n

    if expected_plane == "n":
        assert n and not p

    if expected_plane is None:
        assert not p and not n
