""" Frame and angle manipulation utilities. """

import numpy as np
import pint

from eradiate._units import unit_registry as ureg


@ureg.wraps(ret=None, args=("dimensionless", "rad"), strict=False)
def cos_angle_to_direction(cos_theta, phi):
    r"""Convert a zenith cosine and azimuth angle pair to a direction.

    Parameter ``theta`` (float):
        Zenith angle cosine [dimensionless].
        Convention: 1 corresponds to zenith, -1 corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → array[float]:
        Direction corresponding to the angular parameters [unitless].
    """
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    return np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])


@ureg.wraps(ret=None, args=("rad", "rad"), strict=False)
def angles_to_direction(theta, phi):
    r"""Convert a zenith and azimuth angle pair to a direction.

    Parameter ``theta`` (float):
        Zenith angle [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → array:
        Direction corresponding to the angular parameters [unitless].
    """
    return cos_angle_to_direction(np.cos(theta), phi)


@ureg.wraps(ret="rad", args=None, strict=False)
def direction_to_angles(wi):
    """Converts a cartesian 3-vector to a pair of theta and phi values
    in spherical coordinates

    Parameter ``wi`` (array):
        3-vector designating a direction in cartesian coordinates [unitless].

    Returns → :class:`pint.Quantity`:
        2-vector containing zenith and azimuth angles, where zenith = 0
        corresponds to +z direction [rad].
    """
    wi = wi / np.linalg.norm(wi)
    theta = np.arccos(wi[2])
    phi = np.arctan2(wi[1], wi[0])

    return [theta, phi]


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


# -- Utility warping functions (port to numpy from Mitsuba 2) ------------------


