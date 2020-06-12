""" This module contains frame and angle manipulation utilities. """

import numpy as np


def cos_angle_to_direction(cos_theta, phi):
    r"""Convert a zenith cosine and azimuth angle pair to a direction.

    Parameter ``theta`` (float):
        Zenith angle cosine [dimensionless].
        Convention: 1 corresponds to zenith, -1 corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → array:
        Direction corresponding to the angular parameters.
    """
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    return np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])


def angles_to_direction(theta, phi):
    r"""Convert a zenith and azimuth angle pair to a direction.

    Parameter ``theta`` (float):
        Zenith angle [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → array:
        Direction corresponding to the angular parameters.
    """
    return cos_angle_to_direction(np.cos(theta), phi)


def direction_to_angles(wi):
    """Converts a cartesian 3-vector to a pair of theta and phi values
    in spherical coordinates

    Parameter ``wi`` (array):
        3-vector designating a direction in cartesian coordinates

    Returns → list[float]:
        Zenith and azimuth angles in radians, where zenith=0 corresponds to
        +z direction
    """
    wi = wi / np.linalg.norm(wi)
    theta = np.arccos(wi[2])
    phi = np.arctan2(wi[1], wi[0])

    return [theta, phi]

def spherical_to_cartesian(r, theta, phi):
    r"""Convert spherical coordinates to cartesian coordinates

    Parameter ``r`` (float):
        Radial distance coordinate.

    Parameter ``theta`` (float):
        Zenith angle coordinate [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    Parameter ``phi`` (float):
        Azimuth angle coordinate [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → tuple:
        Cartesian coordinates x, y, z.
    """

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z