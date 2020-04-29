from abc import ABC, abstractmethod

import attr
import numpy as np
from scipy.constants import physical_constants

from .builder import *
from ..util import Q_

# Physical constants
_LOSCHMIDT = Q_(
    *physical_constants['Loschmidt constant (273.15 K, 101.325 kPa)'][:2])


@attr.s
class Atmosphere(ABC):
    """An abstract base class defining common facilities for all atmospheres."""

    @abstractmethod
    def phase(self):
        """Return phase function plugin interfaces.

        return (list): List of ``Phase`` plugin interfaces.
        """
        pass

    @abstractmethod
    def media(self):
        """Return participating media plugin interfaces.

        return (list): List of ``Medium`` plugin interfaces.
        """
        pass

    @abstractmethod
    def shapes(self):
        """Return shape plugin interfaces using references.

        return (list): List of ``Shape`` plugin interfaces.
        """
        pass


def king_correction_factor(ratio=0.0279):
    """
    Compute the King correction factor.

    :param (float) ratio: depolarisation ratio [dimensionless].
        The default value is the mean depolarisation ratio for dry air given by
        Young (1980), Applied Optics, Volume 19, Number 20.

    :return (float): King correction factor [dimensionless].
    """

    return (6. + 3. * ratio) / (6. - 7. * ratio)


def rayleigh_scattering_coefficient_1(
        wavelength=550.,
        number_density=_LOSCHMIDT.magnitude,
        refractive_index=1.0002932,
        king_factor=1.049,
        depolarisation_ratio=None
):
    """Compute the Rayleigh scattering coefficient for one type of scattering
    particles.

    When default values are used, this provides the Rayleigh scattering
    coefficient of the air at 550 nm and under standard temperature and
    pressure conditions.

    :param (float) wavelength: wavelength [nm].
    :param (float) number_density: number density of the scattering particles
        [m^-3].
    :param (float) refractive_index: refractive index of scattering particles
        [dimensionless]. Default value is the air refractive index at 550 nm as
        given by Bates (1984) - Planetary and Space Science, Volume 32, No. 6.
    :param (float) king_factor: King correction factor of the scattering
        particles [dimensionless]. Default value is the
        air effective King factor at 550 nm as given by Bates (1984).
    :param (float) depolarisation_ratio: depolarisation ratio [dimensionless].
        If this parameter is set, then its value is used to compute the value of
        the corresponding King factor and supersedes ``king_factor``.

    :return (float): scattering coefficient in inverse meters.
    """
    if depolarisation_ratio is not None:
        king_factor = king_correction_factor(depolarisation_ratio)

    return \
        8. * np.power(np.pi, 3) / (3. * np.power((wavelength * 1e-9), 4)) / \
        number_density * \
        np.square(np.square(refractive_index) - 1) * king_factor


def rayleigh_scattering_coefficient_mixture(
        wavelength,
        number_densities,
        mixture_refractive_index,
        standard_number_densities,
        standard_refractive_indices,
        king_factors,
        depolarisation_ratios=None
):
    """Compute the Rayleigh scattering coefficient for a mixture of scattering
    particles.

    :param (float or array) wavelength: wavelength [nm].
    :param (array) number_densities: scattering particles number
        densities [m^-3].
    :param (float) mixture_refractive_index: mixture refractive index at the
        specified wavelength [dimensionless].
    :param (array) standard_number_densities: scattering particles
        standard number densities [m^-3].
    :param (array) standard_refractive_indices: scattering particles
        refractive indices at the particles standard number densities
        [dimensionless].
    :param (array) king_factors: King correction factors of the
        scattering particles at the specified wavelength [dimensionless].
    :param (array) depolarisation_ratios: scattering particles
        depolarisation ratios at the specified wavelength. If this parameter is
        set, then the values are used to compute the values of the corresponding
        King factors.

    :return (float or array): mixture Rayleigh scattering coefficient
        [m^-1].
    """

    if depolarisation_ratios is not None:
        king_factors = king_correction_factor(depolarisation_ratios)

    lorenz_factor = (np.square(mixture_refractive_index) + 2.) / 3.

    particles_sum = np.sum(
        (number_densities / np.square(standard_number_densities)) *
        np.square(
            (np.square(standard_refractive_indices) - 1.) /
            (np.square(standard_refractive_indices) + 2.)
        ) * king_factors)

    return \
        24. * np.power(np.pi, 3) / np.power(wavelength * 1e-9, 4) * \
        np.square(lorenz_factor) * particles_sum


def rayleigh_delta(ratio=0.0279):
    """
    Compute the unnamed parameter of the Rayleigh phase function

    :param (float) ratio: depolarisation ratio [dimensionless]
        The default value is the mean depolarisation ratio for dry air given by
        Young (1980), Applied Optics, Volume 19, Number 20.

    :return (float): large delta [dimensionless]
    """
    return (1. - ratio) / (1. + ratio / 2.)


@attr.s()
class RayleighHomogeneous(Atmosphere):
    """This class builds an atmosphere consisting of a non-absorbing homogeneous
    medium. Scattering uses the Rayleigh phase function.
    """

    # Class attributes
    albedo = Spectrum(1.)

    # Instance attributes
    sigma_t = attr.ib(default=rayleigh_scattering_coefficient_1(),
                      converter=Spectrum)

    def __attrs_post_init__(self):
        self.init()

    def init(self):
        """(Re)initialise hidden internal state.
        """
        self._phase = None
        self._medium = None
        self._shapes = None

    def phase(self):
        if self._phase is None:
            self._phase = [phase.Rayleigh(id="phase_rayleigh")]

        return self._phase

    def media(self):
        if self._medium is None:
            phase = self.phase()[0]

            self._medium = [media.Homogeneous(
                id="medium_rayleigh",
                phase=phase.get_ref(),
                sigma_t=self.sigma_t,
                albedo=self.albedo
            )]

        return self._medium

    def shapes(self):
        if self._shapes is None:
            medium = self.media()[0]

            self._shapes = [shapes.Cube(
                to_world=Transform([
                    Scale(value=[1., 1., 1.]),
                    Translate(value=[0., 0., 1.])
                ]),
                bsdf=bsdfs.Null(),
                interior=medium.get_ref()
            )]

        return self._shapes
