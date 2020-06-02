import attr
import numpy as np
from scipy.constants import physical_constants

from .base import Atmosphere
from ...util.units import Q_

offset = {
    "scalar_rgb": 1e-4,
    "scalar_mono": 1e-4,
    "scalar_rgb_double": 1e-7,
    "scalar_mono_double": 1e-7
}

# Physical constants
_LOSCHMIDT = Q_(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2])


def king_factor(ratio=0.0279):
    """Compute the King correction factor.

    Parameter ``ratio`` (float):
        Depolarisation ratio [dimensionless].
        The default value is the mean depolarisation ratio for dry air given by
        :cite:`Young1980RevisedDepolarizationCorrections`.

    Returns → float:
        King correction factor [dimensionless].
    """

    return (6. + 3. * ratio) / (6. - 7. * ratio)


def sigmas_single(
        wavelength=550.,
        number_density=_LOSCHMIDT.magnitude,
        refractive_index=1.0002932,
        king_factor=1.049,
        depolarisation_ratio=None
):
    """Compute the Rayleigh scattering coefficient for one type of scattering
    particles.

    When default values are used, this provides the Rayleigh scattering
    coefficient for air at 550 nm in standard temperature and pressure
    conditions.

    Parameter ``wavelength`` (float):
        Wavelength [nm].

    Parameter ``number_density`` (float):
        Number density of the scattering particles [m^-3].

    Parameter ``refractive_index`` (float):
        Refractive index of scattering particles [dimensionless].
        Default value is the air refractive index at 550 nm as
        given by :cite:`Bates1984RayleighScatteringAir`.

    Parameter ``king_factor`` (float):
        King correction factor of the scattering particles [dimensionless].
        Default value is the air effective King factor at 550 nm as given by
        :cite:`Bates1984RayleighScatteringAir`.

    Parameter ``depolarisation_ratio`` (float):
        Depolarisation ratio [dimensionless].
        If this parameter is set, then its value is used to compute the value of
        the corresponding King factor and supersedes ``king_factor``.

    Returns → float:
        Scattering coefficient [m^1].
    """
    if depolarisation_ratio is not None:
        king_factor = king_factor(depolarisation_ratio)

    return \
        8. * np.power(np.pi, 3) / (3. * np.power((wavelength * 1e-9), 4)) / \
        number_density * \
        np.square(np.square(refractive_index) - 1) * king_factor


def sigmas_mixture(
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

    Parameter ``wavelength`` (float or array):
        Wavelength [nm].

    Parameter ``number_densities`` (array):
        Scattering particles number densities [m^-3].

    Parameter ``mixture_refractive_index`` (float):
        Mixture refractive index at the specified wavelength [dimensionless].

    Parameter ``standard_number_densities`` (array):
        Scattering particles standard number densities [m^-3].

    Parameter ``standard_refractive_indices`` (array):
        scattering particles refractive indices at the particles standard number
        densities [dimensionless].

    Parameter ``king_factors`` (array):
        King correction factors of the scattering particles at the specified
        wavelength [dimensionless].

    Parameter ``depolarisation_ratios`` (array):
        Scattering particles depolarisation ratios at the specified wavelength.
        If this parameter is set, then the values are used to compute the values
        of the corresponding King factors.

    Returns → float or array:
        Mixture Rayleigh scattering coefficient [m^-1].
    """

    if depolarisation_ratios is not None:
        king_factors = king_factor(depolarisation_ratios)

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


def delta(ratio=0.0279):
    r"""Compute the :math:`\Delta` parameter of the Rayleigh phase function.

    Parameter ``ratio`` (float):
        Depolarisation ratio [dimensionless].
        The default value is the mean depolarisation ratio for dry air given by
        :cite:`Young1980RevisedDepolarizationCorrections`.

    Returns → float:
        :math:`\Delta` [dimensionless].
    """
    return (1. - ratio) / (1. + ratio / 2.)


@attr.s()
class RayleighHomogeneous(Atmosphere):
    r"""This class builds an atmosphere consisting of a non-absorbing
    homogeneous medium. Scattering uses the Rayleigh phase function and the
    Rayleigh scattering coefficient of a single gas.

    Constructor arguments / public attributes:
        ``scattering_coefficient`` (float):
            Atmosphere scattering coefficient [m^-1].

        ``rayleigh_parameters`` (dict):
            Parameters of :func:`eradiate.scenes.atmosphere.rayleigh_scattering_coefficient_1`

        ``width`` (float)
            Width of the atmosphere [m].

        ``height`` (float):
            Height of the atmosphere [m].
    """

    # Class attributes
    albedo = 1.

    # Instance attributes
    scattering_coefficient = attr.ib(default=None)
    rayleigh_parameters = attr.ib(default=None)
    width = attr.ib(default=None)
    height = attr.ib(default=1e5)

    def __attrs_post_init__(self):
        self.init()

    def init(self):
        """(Re)initialise hidden internal state."""
        if self.scattering_coefficient is None:
            if self.rayleigh_parameters is None:
                self.rayleigh_parameters = {}
            self.scattering_coefficient = \
                sigmas_single(**self.rayleigh_parameters)
        # TODO: add a warning if both scattering_coefficient and rayleigh_parameters are set

        # if width is not set, compute a value that is large enough (5 times
        # the scattering mean free path) so that there is
        # almost no edge effect
        if self.width is None:
            self.width = 5. / self.scattering_coefficient

    def phase(self):
        return {"phase_atmosphere": {"type": "rayleigh"}}

    def media(self, ref=False):
        if ref:
            phase = {"type": "ref", "id": "phase_atmosphere"}
        else:
            phase = self.phase()["phase_atmosphere"]

        return {
            "medium_atmosphere": {
                "type": "homogeneous",
                "phase": phase,
                "sigma_t": {"type": "uniform", "value": self.scattering_coefficient},
                "albedo": {"type": "uniform", "value": self.albedo},
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f
        from eradiate.kernel import variant

        if ref:
            medium = {"type": "ref", "id": "medium_atmosphere"}
        else:
            medium = self.media(ref=False)["medium_atmosphere"]

        return {
            "shape_atmosphere": {
                "type": "cube",
                "to_world": ScalarTransform4f
                    .scale([self.width / 2., self.width / 2., self.height / 2.])
                    .translate([0., 0., self.height / 2. + offset[variant()]]),
                    # TODO: remove offset dict and replace it with a formula using eradiate.kernel.core.math.Epsilon (or RayEpsilon, or ShadowEpsilon)
                "bsdf": {"type": "null"},
                "interior": medium
            }
        }
