import attr
import numpy as np
from scipy.constants import physical_constants

from .base import Atmosphere
from ..factory import Factory
from ...util.units import Q_

offset = {
    "scalar_rgb": 1e-4,
    "scalar_mono": 1e-4,
    "scalar_rgb_double": 1e-7,
    "scalar_mono_double": 1e-7
}

# Physical constants
#: Loschmidt constant [m^-3].
_LOSCHMIDT = Q_(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2])
#: Refractive index of dry air [dimensionless].
_IOR_DRY_AIR = Q_(1.0002932, "")


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
    refractive_index=_IOR_DRY_AIR.magnitude,
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
        np.square(np.square(refractive_index) - 1.) * king_factor


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
@Factory.register("rayleigh_homogeneous")
class RayleighHomogeneous(Atmosphere):
    r"""This class builds an atmosphere consisting of a non-absorbing
    homogeneous medium. Scattering uses the Rayleigh phase function and the
    Rayleigh scattering coefficient of a single gas.

    TODO: update docs

    Constructor arguments / public attributes:
        ``height`` (float):
            Height of the atmosphere [m].

        ``width`` (float)
            Width of the atmosphere [m].

        ``scattering_coefficient`` (float):
            Atmosphere scattering coefficient [m^-1].

        ``wavelength`` (float):
            Wavelength [nm].

        ``number_density`` (float):
            Number density of the scattering particles [m^-3].

        ``refractive_index`` (float):
            Refractive index of scattering particles [dimensionless].
            Default value is the air refractive index at 550 nm as
            given by :cite:`Bates1984RayleighScatteringAir`.

        ``king_factor`` (float):
            King correction factor of the scattering particles [dimensionless].
            Default value is the air effective King factor at 550 nm as given by
            :cite:`Bates1984RayleighScatteringAir`.

        ``depolarisation_ratio`` (float):
            Depolarisation ratio [dimensionless].
            If this parameter is set, then its value is used to compute the value of
            the corresponding King factor and supersedes ``king_factor``.

        Note: If ``scattering_coefficient`` is set, ``wavelength``,
        ``number_density``, ``refractive_index``, ``king_factor`` and
        ``depolarisation_ratio`` must not be set. If ``scattering_coefficient``
        is not set, ``wavelength``, ``number_density``, ``refractive_index``,
        and ``king_factor`` or ``depolarisation_ratio`` are used to compute the
        scattering coefficient.
    """

    # Class attributes
    DEFAULT_CONFIG = {
        "height": 1e5,
        # "width": None,
        "sigmas": 1e-6,
        # "sigmas_params": None
    }
    ALBEDO = 1.

    def init(self):
        r"""(Re)initialise hidden internal state.
        """
        print(self.config)
        # If sigmas_params is set, override sigmas based on parametrisation
        if self.config.get("sigmas_params", None) is not None:
            self.config["sigmas"] = \
                sigmas_single(**self.config["sigmas_params"])

        # If width is not set, compute a value corresponding to an optically
        # thick layer (10x scattering mean free path)
        if self.config.get("width", None) is None:
            self.config["width"] = 10. / self.config["sigmas"]

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
                "sigma_t": {"type": "uniform", "value": self.config["sigmas"]},
                "albedo": {"type": "uniform", "value": self.ALBEDO},
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f
        from eradiate.kernel.core.math import ShadowEpsilon
        offset = ShadowEpsilon

        if ref:
            medium = {"type": "ref", "id": "medium_atmosphere"}
        else:
            medium = self.media(ref=False)["medium_atmosphere"]

        width = self.config["width"]
        height = self.config["height"]

        return {
            "shape_atmosphere": {
                "type": "cube",
                "to_world": ScalarTransform4f
                    .scale([0.5 * width, 0.5 * width, 0.5 * height])
                    .translate([0.0, 0.0, 0.5 * height + offset]),
                "bsdf": {"type": "null"},
                "interior": medium
            }
        }
