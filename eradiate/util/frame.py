""" This module contains frame and angle manipulation utilities. """

import numpy as np


def cos_angle_to_direction(cos_theta, phi):
    r"""Convert a zenith cosine and azimuth angle pair to a direction.

    Parameter ``cos_theta`` (float)
        Zenith angle cosine [dimensionless].
        Convention: 1 corresponds to zenith, -1 corresponds to nadir.

    Parameter ``phi`` (float)
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → numpy.ndarray
        Direction corresponding to the angular parameters.
    """
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    return np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])


def angles_to_direction(theta, phi):
    r"""Convert a zenith and azimuth angle pair to a direction.

    Parameter ``theta`` (float)
        Zenith angle [radian].
        Convention: 0 corresponds to zenith, :math:`\pi` corresponds to nadir.

    Parameter ``phi`` (float)
        Azimuth angle [radian].
        Convention: :math:`2 \pi` corresponds to the X axis.

    Returns → numpy.ndarray
        Direction corresponding to the angular parameters.
    """
    return cos_angle_to_direction(np.cos(theta), phi)
