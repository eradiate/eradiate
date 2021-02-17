import numpy as np

from eradiate import unit_registry as ureg
from eradiate.frame import (
    angles_to_direction,
    cos_angle_to_direction,
    direction_to_angles,
    spherical_to_cartesian
)


def test_cos_angle_to_direction():
    assert np.allclose(cos_angle_to_direction(1., 0.), [0, 0, 1])
    assert np.allclose(cos_angle_to_direction(0., 0.), [1, 0, 0])
    assert np.allclose(cos_angle_to_direction(-1., ureg.Quantity(135., "deg")), [0, 0, -1])


def test_angles_to_direction():
    assert np.allclose(angles_to_direction(0., 0.), [0, 0, 1])
    assert np.allclose(angles_to_direction(0.5 * np.pi, 0.), [1, 0, 0])
    assert np.allclose(
        angles_to_direction(ureg.Quantity(45.), ureg.Quantity(135., "deg")),
        [-0.60167965,  0.60167965,  0.52532199]
    )


def test_direction_to_angles():
    assert np.allclose(direction_to_angles([0, 0, 1]), (0, 0))
    assert np.allclose(
        direction_to_angles([np.sqrt(2) / 2, np.sqrt(2) / 2, 0]),
        (ureg.Quantity(90., "deg"), ureg.Quantity(45., "deg"))
    )


def test_spherical_to_cartesian():
    r = 2.
    theta = np.deg2rad(30)
    phi = np.deg2rad(0)
    d = spherical_to_cartesian(r, theta, phi)
    assert np.allclose(d, [1, 0, np.sqrt(3)])

    r = ureg.Quantity(2., "km")
    theta = np.deg2rad(60)
    phi = np.deg2rad(30)
    d = spherical_to_cartesian(r, theta, phi)
    assert np.allclose(d, ureg.Quantity([3. / 2., np.sqrt(3) / 2., 1.], "km"))
