""" Frame and angle manipulation utilities. """

import numpy as np
import pint
from numpy.typing import ArrayLike

from ._units import unit_registry as ureg
from ._util import ensure_array


@ureg.wraps(ret=None, args=("dimensionless", "rad"), strict=False)
def cos_angle_to_direction(cos_theta, phi):
    r"""
    Convert a zenith cosine and azimuth angle pair to a direction.

    Parameter ``theta`` (float):
        Zenith angle cosine [dimensionless].
        Convention: 1 corresponds to zenith, -1 corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → array[float]:
        Direction corresponding to the angular parameters [unitless].
    """
    cos_theta = ensure_array(cos_theta)
    phi = ensure_array(phi)

    sin_theta = np.sqrt(1.0 - np.multiply(cos_theta, cos_theta))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    return np.vstack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)).T


@ureg.wraps(ret=None, args=("rad",), strict=False)
def angles_to_direction(angles):
    r"""
    Convert a zenith and azimuth angle pair to a direction.

    Parameter ``theta`` (float):
        Zenith angle [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → array:
        Direction corresponding to the angular parameters [unitless].
    """
    angles = ensure_array(angles)
    if angles.ndim < 2:
        angles = angles.reshape((angles.size // 2, 2))
    if angles.ndim > 2 or angles.shape[1] != 2:
        raise ValueError(f"array must be of shape (N, 2), got {angles.shape}")

    return cos_angle_to_direction(np.cos(angles[:, 0]), angles[:, 1])


@ureg.wraps(ret="rad", args=None, strict=False)
def direction_to_angles(v: ArrayLike) -> ArrayLike:
    """
    Convert a cartesian unit vector to a zenith-azimuth pair.

    Parameter ``v`` (array-like):
        A sequence of 3-vectors (shape (N, 3)) [unitless].

    Returns → array-like:
        A sequence of 2-vectors containing zenith and azimuth angles, where
        zenith = 0 corresponds to +z direction (shape (N, 2)) [rad].
    """
    v = ensure_array(v)
    if v.ndim < 2:
        v = v.reshape((v.size // 3, 3))
    if v.ndim > 2 or v.shape[1] != 3:
        raise ValueError(f"array must be of shape (N, 3), got {v.shape}")

    v = v / np.linalg.norm(v, axis=-1).reshape(len(v), 1)
    theta = np.arccos(v[..., 2])
    phi = np.arctan2(v[..., 1], v[..., 0])

    return np.vstack((theta, phi)).T


@ureg.wraps(ret=None, args=(None, "rad", "rad", None), strict=False)
def spherical_to_cartesian(r, theta, phi, origin=np.zeros((3,))):
    r"""Convert spherical coordinates to cartesian coordinates

    Parameter ``r`` (float):
        Radial distance coordinate.

    Parameter ``theta`` (float):
        Zenith angle coordinate [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle coordinate [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Parameter ``origin`` (array):
        Shifts the center point of the coordinate system.

    Returns → array[float]:
        Cartesian coordinates x, y, z.
    """
    # TODO: Vectorise

    # fmt: off
    if isinstance(r, pint.Quantity):
        return np.array([
            r.magnitude * np.sin(theta) * np.cos(phi) + origin[0],
            r.magnitude * np.sin(theta) * np.sin(phi) + origin[1],
            r.magnitude * np.cos(theta) + origin[2]
        ]) * r.units
    else:
        return np.array([
            r * np.sin(theta) * np.cos(phi) + origin[0],
            r * np.sin(theta) * np.sin(phi) + origin[1],
            r * np.cos(theta) + origin[2]
        ])
    # fmt: on
