"""Frame and angle manipulation utilities."""

import numpy as np
import pint
from numpy.typing import ArrayLike

from .units import unit_registry as ureg


@ureg.wraps(ret=None, args=("dimensionless", "rad"), strict=False)
def cos_angle_to_direction(cos_theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    r"""
    Convert a zenith cosine and azimuth angle pair to a direction.

    Parameters
    ----------
    theta : array-like
        Zenith angle cosine [dimensionless].
        Convention: 1 corresponds to zenith, -1 corresponds to nadir.

    phi : array-like
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns
    -------
    ndarray
        Direction corresponding to the angular parameters [unitless].
    """
    cos_theta = np.atleast_1d(cos_theta)
    phi = np.atleast_1d(phi)

    sin_theta = np.sqrt(1.0 - np.multiply(cos_theta, cos_theta))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    return np.vstack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)).T


@ureg.wraps(ret=None, args=("rad",), strict=False)
def angles_to_direction(angles: ArrayLike) -> np.ndarray:
    r"""
    Convert a zenith and azimuth angle pair to a direction unit vector.

    Parameters
    ----------
    theta : array-like
        Zenith angle [radian].
        0 corresponds to zenith, :math:`\pi/2` corresponds to the XY plane,
        :math:`\pi` corresponds to nadir.
        Negative values are allowed; :math:`(\theta, \varphi)`
        then maps to :math:`(| \theta |, \varphi + \pi)`.

    phi : array-like
        Azimuth angle [radian].
        0 corresponds to the X axis, :math:`\pi / 2` corresponds to the Y axis
        (*i.e.* rotation is counter-clockwise).

    Returns
    -------
    ndarray
        Direction corresponding to the angular parameters [unitless].
    """
    angles = np.atleast_1d(angles)
    if angles.ndim < 2:
        angles = angles.reshape((angles.size // 2, 2))
    if angles.ndim > 2 or angles.shape[1] != 2:
        raise ValueError(f"array must be of shape (N, 2), got {angles.shape}")

    negative_zenith = angles[:, 0] < 0
    angles[negative_zenith, 0] *= -1
    angles[negative_zenith, 1] += np.pi

    return cos_angle_to_direction(np.cos(angles[:, 0]), angles[:, 1])


@ureg.wraps(ret="rad", args=None, strict=False)
def direction_to_angles(v: ArrayLike) -> np.ndarray:
    """
    Convert a cartesian unit vector to a zenith-azimuth pair.

    Parameters
    ----------
    v : array-like
        A sequence of 3-vectors (shape (N, 3)) [unitless].

    Returns
    -------
    array-like
        A sequence of 2-vectors containing zenith and azimuth angles, where
        zenith = 0 corresponds to +z direction (shape (N, 2)) [rad].
    """
    v = np.atleast_1d(v)
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
    r"""
    Convert spherical coordinates to cartesian coordinates

    Parameters
    ----------
    r : float
        Radial distance coordinate.

    theta : float
        Zenith angle coordinate [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    phi : float
        Azimuth angle coordinate [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    origin : array-like
        Shifts the center point of the coordinate system.

    Returns
    -------
    ndarray
        Cartesian coordinates x, y, z as a (3,) array.
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


def angles_in_hplane(
    plane: float,
    theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
    raise_exc: bool = True,
):
    """
    Check that a set of (zenith, azimuth) pairs belong to a given hemisphere
    plane cut.

    Parameters
    ----------
    plane : float
        Plane cut orientation in degrees.

    theta : ndarray
        List of zenith angle values with (N,) shape in degrees.

    phi : ndarray
        List of azimuth angle values with (N,) shape in degrees.

    raise_exc : bool, optional
        If ``True``, raise if not all directions are snapped to the specified
        hemisphere plane cut.

    Returns
    -------
    in_plane_positive, in_plane_negative
        Masks indicating indexes of directions contained in the positive (resp.
        negative) half-plane.

    Raises
    ------
    ValueError
        If not all directions are snapped to the specified hemisphere plane cut.
    """
    # Normalise input parameters
    phi = np.where(theta >= 0.0, phi % 360, (phi + 180) % 360)
    theta = np.where(theta >= 0.0, theta, -theta)

    # Assign angle pairs to positive or negative half-plane
    in_plane_positive = np.isclose(plane, phi) | np.isclose(0.0, theta)
    in_plane_negative = np.isclose((plane + 180) % 360, phi) & ~in_plane_positive
    in_plane = in_plane_positive | in_plane_negative

    if raise_exc and not (np.all(in_plane)):
        raise ValueError("found off-plane directions")

    return in_plane_positive, in_plane_negative
