"""Functions to compute Rayleigh scattering in the air.
"""

import numpy as np
from scipy.constants import physical_constants

from .._units import unit_registry as ureg

# Physical constants
#: Loschmidt constant [km^-3].
_LOSCHMIDT = ureg.Quantity(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2]).to(
    "km^-3")

# Air number density at 101325 Pa and 288.15 K
_STANDARD_AIR_NUMBER_DENSITY = _LOSCHMIDT * (273.15 / 288.15)


def kf(ratio=0.0279):
    """Compute the King correction factor.

    Parameter ``ratio`` (float):
        Depolarisation ratio [dimensionless].
        The default value is the mean depolarisation ratio for dry air given by
        :cite:`Young1980RevisedDepolarizationCorrections`.

    Returns → float:
        King correction factor [dimensionless].
    """

    return (6.0 + 3.0 * ratio) / (6.0 - 7.0 * ratio)


@ureg.wraps(ret="km^-1", args=("nm", "km^-3", None, None), strict=False)
def compute_sigma_s_air(wavelength=550.,
                        number_density=_STANDARD_AIR_NUMBER_DENSITY.magnitude,
                        king_factor=1.049,
                        depolarisation_ratio=None):
    """Compute the Rayleigh scattering coefficient of air.

    When default values are used, this provides the Rayleigh scattering
    coefficient for air at 550 nm in standard temperature and pressure
    conditions.

    Parameter ``wavelength`` (float):
        Wavelength [nm].

    Parameter ``number_density`` (float):
        Number density of the scattering particles [km^-3].

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

    refractive_index = air_refractive_index(
        wavelength=wavelength,
        number_density=ureg.Quantity(number_density, "km^-3")
    )

    return \
        8. * np.power(np.pi, 3) / (3. * np.power((wavelength * 1e-12), 4)) / \
        number_density * \
        np.square(np.square(refractive_index) - 1.) * king_factor


@ureg.wraps(ret=None, args=("nanometer", "m^-3"), strict=False)
def air_refractive_index(wavelength=550.,
                         number_density=_STANDARD_AIR_NUMBER_DENSITY.to("m^-3").magnitude):
    """Computes the air refractive index.

    The wavelength dependence of the refractive index is computed using equation
    2 from :cite:`Peck1972DispersionAir`. This formula is a fit of
    measurements of the air refractive index in the range
    The number density dependence is computed using a simple proportionality
    rule.

    Parameter ``wavelength`` (float or array):
        Wavelength value [nanometer].

        Default: 550.

    Parameter ``number_density`` (float or array):
        Number density [m^-3].

        Default: Air number density at 101325 Pa and 288.15 K.

    Returns → float or array:
        Air refractive index value(s).
    """

    # wavenumber in inverse micrometer
    sigma = 1 / ureg.Quantity(wavelength, "nm").to("micrometer").magnitude
    sigma2 = np.square(sigma)

    # refractivity in parts per 1e8
    x = (5791817. / (238.0183 - sigma2)) + 167909. / (57.362 - sigma2)

    # number density scaling
    x *= number_density / _STANDARD_AIR_NUMBER_DENSITY.to("m^-3").magnitude

    # refractive index
    index = 1 + x * 1e-8

    return index
