"""Frame and angle manipulation utilities."""

from __future__ import annotations

import enum
import typing as t

import aenum
import numpy as np
import pint

from .units import unit_registry as ureg


class AzimuthConvention(enum.Enum):
    """
    An enumeration of azimuth convention names associated with the corresponding
    (origin offset, orientation) pair with respect to the *East right*
    convention. The origin offset is expressed in radian and the orientation is
    the angle value multiplier (±1).
    """

    EAST_RIGHT = (0.0, 1)  #: East right
    EAST_LEFT = (0.0, -1)  #: East left
    NORTH_RIGHT = (0.5 * np.pi, 1)  #: North right
    NORTH_LEFT = (0.5 * np.pi, -1)  #: North left
    WEST_RIGHT = (np.pi, 1)  #: West right
    WEST_LEFT = (np.pi, -1)  #: West left
    SOUTH_RIGHT = (1.5 * np.pi, 1)  #: South right
    SOUTH_LEFT = (1.5 * np.pi, -1)  #: South left

    @staticmethod
    def convert(value: t.Any) -> AzimuthConvention:
        """
        Attempt conversion of a value to an :class:`.AzimuthConvention`
        instance. The conversion protocol is as follows:

        * If ``value`` is a string, it is converted to upper case and passed to
          the indexing operator of :class:`.AzimuthConvention`.
        * If ``value`` is an :class:`.AzimuthConvention` instance, it is returned
          without change.
        * Otherwise, the method raises an exception.

        Parameters
        ----------
        value
            Value to attempt conversion of.

        Returns
        -------
        Converted value

        Raises
        ------
        TypeError
            If no conversion protocol exists for ``value``.
        """
        if isinstance(value, str):
            return AzimuthConvention[value.upper()]
        elif isinstance(value, AzimuthConvention):
            return value
        else:
            raise TypeError(
                f"Cannot convert a {type(value)} instance to AzimuthConvention"
            )

    @classmethod
    def register(cls, name: str, value: tuple[float, float]) -> None:
        """
        Register a new angular convention.

        Parameters
        ----------
        name : str
            Name of the registered convention. Should be uppercase, without
            whitespace.

        value : tuple
            An (offset, orientation) 2-tuple. The offset is the angular offset
            of the registered convention in radian. The offset is an integer
            equal to 1 (right-hand/counter-clockwise convention) or -1
            (left-hand/clockwise convention).
        """
        aenum.extend_enum(cls, name.upper(), value)


def normalize_azimuth(angles: np.typing.ArrayLike, inplace: bool = False) -> np.ndarray:
    """
    Normalize azimuth values to the [0, 2π[ interval.

    Parameters
    ----------
    angles : array-like
        A sequence of azimuth values [rad].

    inplace : bool, optional, default: False
        If ``True``, perform the conversion in-place. This will mutate the
        `angles` array; otherwise, this function operates on a copy.

    Returns
    -------
    ndarray
        Azimuth angle values normalized to the [0, 2π[ interval [rad].

    Warnings
    --------
    This function does *not* apply unit conversion automatically: angle values
    must be supplied in radians and are returned as plain Numpy arrays.
    """
    result = angles if inplace else np.copy(angles)
    result %= 2.0 * np.pi

    # Snap close-to-2π values to 0 to compensate for numerical precision-related
    # unwanted shifts (may happen with angles computed from directions)
    snapped = np.where(
        np.isclose(result, 2.0 * np.pi, rtol=0.0, atol=1e-6 * np.pi), 0.0, result
    )

    if inplace:
        result[:] = snapped
    else:
        result = snapped

    return result


def transform_azimuth(
    angles: np.typing.ArrayLike,
    from_convention: AzimuthConvention | str = AzimuthConvention.EAST_RIGHT,
    to_convention: AzimuthConvention | str = AzimuthConvention.EAST_RIGHT,
    normalize: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Convert azimuth values expressed with a given convention to another. The
    East right convention is used as a pivot.

    Parameters
    ----------
    angles : array-like
        A sequence of azimuth values [rad].

    from_convention : .AzimuthConvention or str, optional, default: .AzimuthConvention.EAST_RIGHT
        Source azimuth angle convention. If a string is passed, it will be
        converted to a :class:`.AzimuthConvention`.

    to_convention : .AzimuthConvention or str, optional, default: .AzimuthConvention.EAST_RIGHT
        Target azimuth angle convention. If a string is passed, it will be
        converted to a :class:`.AzimuthConvention`.

    normalize : bool, optional, default: True
        If ``True``, normalize returned angle values within the [0, 2π[
        interval.

    inplace : bool, optional, default: False
        If ``True``, perform the conversion in-place. This will mutate the
        `angles` array; otherwise, this function operates on a copy.

    Returns
    -------
    ndarray
        Azimuth angle values converted to the East right convention [rad].

    Warnings
    --------
    This function does *not* apply unit conversion automatically: angle values
    must be supplied in radians and are returned as plain Numpy arrays.
    """
    result = angles if inplace else np.copy(angles)

    from_convention = AzimuthConvention.convert(from_convention)
    to_convention = AzimuthConvention.convert(to_convention)

    if from_convention is not to_convention:
        from_offset, from_orientation = from_convention.value
        to_offset, to_orientation = to_convention.value

        # Transform to East right
        result *= from_orientation
        result += from_offset

        # Transform to target convention
        result -= to_offset
        result *= to_orientation

    if normalize:
        result = normalize_azimuth(result, inplace=inplace)

    return result


@ureg.wraps(ret=None, args=("dimensionless", "rad", None, None), strict=False)
def cos_angle_to_direction(
    cos_theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
    azimuth_convention: AzimuthConvention | str = AzimuthConvention.EAST_RIGHT,
    flip: bool = False,
) -> np.ndarray:
    r"""
    Convert a zenith cosine and azimuth angle pair to a direction.

    Parameters
    ----------
    cos_theta : array-like
        Zenith angle cosine [dimensionless].
        Convention: 1 corresponds to zenith, -1 corresponds to nadir.

    phi : array-like
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    azimuth_convention : .AzimuthConvention or str, optional, default: .AzimuthConvention.EAST_RIGHT
        Source azimuth angle convention. If a string is passed, it will be
        converted to a :class:`.AzimuthConvention`.

    flip : bool
        If ``True``, flip the returned direction (points towards the nadir with
        `cos_theta` equal to 1).

    Returns
    -------
    ndarray
        Directions corresponding to the angular parameters [unitless].
    """
    cos_theta = np.atleast_1d(cos_theta).astype(float)
    phi = np.atleast_1d(
        transform_azimuth(
            phi,
            from_convention=azimuth_convention,
            to_convention=AzimuthConvention.EAST_RIGHT,
        )
    )

    sin_theta = np.sqrt(1.0 - np.multiply(cos_theta, cos_theta))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    result = np.vstack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)).T
    return result if not flip else -result


@ureg.wraps(ret=None, args=("rad", None, None), strict=False)
def angles_to_direction(
    angles: np.typing.ArrayLike,
    azimuth_convention: AzimuthConvention | str = AzimuthConvention.EAST_RIGHT,
    flip: bool = False,
) -> np.ndarray:
    r"""
    Convert a zenith and azimuth angle pair to a direction unit vector.

    Parameters
    ----------
    angles : array-like
        A sequence of (zenith, azimuth) pairs, where zenith = 0 corresponds to
        +z direction [rad].

    azimuth_convention : .AzimuthConvention or str, optional, default: .AzimuthConvention.EAST_RIGHT
        Source azimuth angle convention. If a string is passed, it will be
        converted to a :class:`.AzimuthConvention`.

    flip : bool, optional, default: False
        If ``True``, flip returned directions (useful in practice to get
        direction vectors pointing towards the local frame origin).

    Returns
    -------
    ndarray
        Directions corresponding to the angular parameters [unitless].
    """
    # Ensure correct shape
    angles = np.atleast_1d(angles).astype(float)
    if angles.ndim < 2:
        angles = angles.reshape((angles.size // 2, 2))
    if angles.ndim > 2 or angles.shape[1] != 2:
        raise ValueError(f"array must be of shape (N, 2), got {angles.shape}")

    # Pre-process zenith
    negative_zenith = angles[:, 0] < 0
    angles[negative_zenith, 0] *= -1
    angles[negative_zenith, 1] += np.pi

    return cos_angle_to_direction(
        np.cos(angles[:, 0]),
        angles[:, 1],
        flip=flip,
        azimuth_convention=azimuth_convention,
    )


@ureg.wraps(ret="rad", args=(None, None, None), strict=False)
def direction_to_angles(
    v: np.typing.ArrayLike,
    azimuth_convention: AzimuthConvention | str = AzimuthConvention.EAST_RIGHT,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert a cartesian unit vector to a zenith-azimuth pair.

    Parameters
    ----------
    v : array-like
        A sequence of 3-vectors (shape (N, 3)) [unitless].

    azimuth_convention : .AzimuthConvention or str, optional, default: .AzimuthConvention.EAST_RIGHT
        Target azimuth angle convention. If a string is passed, it will be
        converted to a :class:`.AzimuthConvention`.

    normalize : bool, optional, default: True
        If ``True``, normalize azimuth values within the [0, 2π[ interval.

    Returns
    -------
    quantity
        A sequence of 2-vectors containing zenith and azimuth angles, where
        zenith = 0 corresponds to +z direction (shape (N, 2)).
    """
    v = np.atleast_1d(v).astype(float)
    if v.ndim < 2:
        v = v.reshape((v.size // 3, 3))
    if v.ndim > 2 or v.shape[1] != 3:
        raise ValueError(f"array must be of shape (N, 3), got {v.shape}")

    v = v / np.linalg.norm(v, axis=-1).reshape(len(v), 1)
    theta = np.arccos(v[..., 2])
    phi = transform_azimuth(
        np.arctan2(v[..., 1], v[..., 0]),
        from_convention=AzimuthConvention.EAST_RIGHT,
        to_convention=azimuth_convention,
        normalize=normalize,
    )

    return np.vstack((theta, phi)).T


@ureg.wraps(ret=None, args=(None, "rad", "rad", None), strict=False)
def spherical_to_cartesian(r, theta, phi, origin=(0, 0, 0)):
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
        Plane cut orientation [rad].

    theta : ndarray
        List of zenith angle values with (N,) shape [rad].

    phi : ndarray
        List of azimuth angle values with (N,) shape [rad].

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
    # Normalize input parameters
    twopi = 2.0 * np.pi
    phi = np.where(theta >= 0.0, phi % twopi, (phi + np.pi) % twopi)
    theta = np.where(theta >= 0.0, theta, -theta)

    # Assign angle pairs to positive or negative half-plane
    in_plane_positive = np.isclose(plane, phi) | np.isclose(0.0, theta)
    in_plane_negative = np.isclose((plane + np.pi) % twopi, phi) & ~in_plane_positive
    in_plane = in_plane_positive | in_plane_negative

    if raise_exc and not (np.all(in_plane)):
        raise ValueError("found off-plane directions")

    return in_plane_positive, in_plane_negative
