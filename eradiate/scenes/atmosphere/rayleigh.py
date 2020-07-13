import attr
import numpy as np
from scipy.constants import physical_constants

from .base import Atmosphere
from ..core import Factory
from ...util.collections import frozendict
from ...util.units import Q_

# Physical constants
#: Loschmidt constant [km^-3].
_LOSCHMIDT = Q_(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2]).to(
        "km^-3")
#: Refractive index of dry air [dimensionless].
_IOR_DRY_AIR = Q_(1.0002932, "")


def kf(ratio=0.0279):
    """Compute the King correction factor.

    Parameter ``ratio`` (float):
        Depolarisation ratio [dimensionless].
        The default value is the mean depolarisation ratio for dry air given by
        :cite:`Young1980RevisedDepolarizationCorrections`.

    Returns → float:
        King correction factor [dimensionless].
    """

    return (6. + 3. * ratio) / (6. - 7. * ratio)


def sigma_s_single(wavelength=550.,
                   number_density=_LOSCHMIDT.magnitude,
                   refractive_index=_IOR_DRY_AIR.magnitude,
                   king_factor=1.049,
                   depolarisation_ratio=None):
    """Compute the Rayleigh scattering coefficient for one type of scattering
    particles.

    When default values are used, this provides the Rayleigh scattering
    coefficient for air at 550 nm in standard temperature and pressure
    conditions.

    Parameter ``wavelength`` (float):
        Wavelength [nm].

    Parameter ``number_density`` (float):
        Number density of the scattering particles [km^-3].

    Parameter ``refractive_index`` (float):
        Refractive index of scattering particles [dimensionless].
        Default value is the air refractive index at 550 nm as
        given by :cite:`Bates1984RayleighScatteringAir`.

    Parameter ``king_factor`` (float):
        King correction factor of the scattering particles [dimensionless].
        Default value is the air effective King factor at 550 nm as given by
        :cite:`Bates1984RayleighScatteringAir`. Overridden by a call to
        :func:`kf` if ``depolarisation_ratio`` is set.

    Parameter ``depolarisation_ratio`` (float or None):
        Depolarisation ratio [dimensionless].
        If this parameter is set, then its value is used to compute the value of
        the corresponding King factor and supersedes ``king_factor``.

    Returns → float:
        Scattering coefficient [km^-1].
    """
    if depolarisation_ratio is not None:
        king_factor = kf(depolarisation_ratio)

    return \
        8. * np.power(np.pi, 3) / (3. * np.power((wavelength * 1e-12), 4)) / \
        number_density * \
        np.square(np.square(refractive_index) - 1.) * king_factor


def sigma_s_mixture(wavelength,
                    number_densities,
                    mixture_refractive_index,
                    standard_number_densities,
                    standard_refractive_indices,
                    king_factors,
                    depolarisation_ratios=None):
    """Compute the Rayleigh scattering coefficient for a mixture of scattering
    particles.

    Parameter ``wavelength`` (float or array):
        Wavelength [nm].

    Parameter ``number_densities`` (array):
        Scattering particles number densities [km^-3].

    Parameter ``mixture_refractive_index`` (float):
        Mixture refractive index at the specified wavelength [dimensionless].

    Parameter ``standard_number_densities`` (array):
        Scattering particles standard number densities [km^-3].

    Parameter ``standard_refractive_indices`` (array):
        scattering particles refractive indices at the particles standard number
        densities [dimensionless].

    Parameter ``king_factors`` (array):
        King correction factors of the scattering particles at the specified
        wavelength [dimensionless]. Overridden by a call to :func:`kf` if
        ``depolarisation_ratio`` is set.

    Parameter ``depolarisation_ratios`` (array):
        Scattering particles depolarisation ratios at the specified wavelength.
        If this parameter is set, then the values are used to compute the values
        of the corresponding King factors.

    Returns → float or array:
        Mixture Rayleigh scattering coefficient [km^-1].
    """

    if depolarisation_ratios is not None:
        king_factors = kf(depolarisation_ratios)

    lorenz_factor = (np.square(mixture_refractive_index) + 2.) / 3.

    particles_sum = np.sum(
        (number_densities / np.square(standard_number_densities)) * np.square(
            (np.square(standard_refractive_indices) - 1.) /
            (np.square(standard_refractive_indices) + 2.)) * king_factors)

    return \
        24. * np.power(np.pi, 3) / np.power(wavelength * 1e-12, 4) * \
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
class RayleighHomogeneousAtmosphere(Atmosphere):
    r"""Rayleigh homogeneous atmosphere scene generation helper [:factorykey:`rayleigh_homogeneous`].

    This class builds an atmosphere consisting of a non-absorbing
    homogeneous medium. Scattering uses the Rayleigh phase function and the
    Rayleigh scattering coefficient of a single gas.

    .. admonition:: Configuration format
        :class: hint

        ``height`` (float):
            Height of the atmosphere [km].

            Default: 100.

        ``width`` (float or string)
            Width of the atmosphere [km].
            If the string ``"auto"`` is passed, a value will be estimated to
            ensure that the medium is optically thick.

            Default: None.

        ``sigma_s`` (float or dict):
            Atmosphere scattering coefficient value [km^-1] or keyword argument
            dictionary to be passed to
            :func:`~eradiate.scenes.atmosphere.rayleigh.sigma_s_single`.
            If a dictionary is passed and misses arguments,
            :func:`~eradiate.scenes.atmosphere.rayleigh.sigma_s_single`'s
            defaults apply as usual.

            Default: {}.
    """

    # Class attributes
    CONFIG_SCHEMA = frozendict({
        "height": {
            "type": "number",
            "min": 0.,
            "default": 1.e+2
        },
        "width": {
            "anyof": [{
                "type": "number",
                "min": 0.
            }, {
                "type": "string",
                "allowed": ["auto"]
            }],
            "default": "auto"
        },
        "sigma_s": {
            "oneof": [{
                "type": "number",
                "min": 0.,
            }, {
                "type": "dict",
                "schema": {
                    "wavelength": {
                        "type": "number",
                        "min": 0.0,
                    },
                    "number_density": {
                        "type": "number",
                        "min": 0.0,
                    },
                    "refractive_index": {
                        "type": "number",
                        "min": 0.0,
                    },
                    "king_factor": {
                        "type": "number",
                        "min": 0.0,
                    },
                    "depolarisation_ratio": {
                        "type": "number",
                        "min": 0.0,
                        "nullable": True,
                    },
                }
            }],
            "default": {},
        },
    })

    @property
    def _albedo(self):
        """Return albedo."""
        return 1.

    @property
    def _width(self):
        """Return scene width based on configuration."""
        # TODO: make this a cached property

        # If width is not set, compute a value corresponding to an optically
        # thick layer (10x scattering mean free path)
        width = self.config["width"]

        if width == "auto":
            return 10. / self._sigma_s
        else:
            return width

    @property
    def _sigma_s(self):
        """Return scattering coefficient based on configuration."""
        # TODO: make this a cached property

        if isinstance(self.config["sigma_s"], dict):
            return sigma_s_single(**self.config["sigma_s"])
        else:
            return self.config["sigma_s"]

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
                "sigma_t": {
                    "type": "uniform",
                    "value": self._sigma_s
                },
                "albedo": {
                    "type": "uniform",
                    "value": self._albedo
                },
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        if ref:
            medium = {"type": "ref", "id": "medium_atmosphere"}
        else:
            medium = self.media(ref=False)["medium_atmosphere"]

        width = self._width
        height = self.config["height"]
        height_offset = height * 0.01

        return {
            "shape_atmosphere": {
                "type":
                    "cube",
                "to_world":
                    ScalarTransform4f([
                        [0.5 * width, 0., 0., 0.],
                        [0., 0.5 * width, 0., 0.],
                        [
                            0., 0., 0.5 * (height + height_offset),
                            0.5 * (height - height_offset)
                        ],
                        [0., 0., 0., 1.],
                    ]),
                "bsdf": {
                    "type": "null"
                },
                "interior":
                    medium
            }
        }
