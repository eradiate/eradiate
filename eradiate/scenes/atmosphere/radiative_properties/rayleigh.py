import numpy as np
from scipy.constants import physical_constants

from ....util.units import ureg

_Q = ureg.Quantity


# Physical constants
#: Loschmidt constant [km^-3].
_LOSCHMIDT = ureg.Quantity(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2]).to(
    "km^-3")
#: Refractive index of dry air [dimensionless].
_IOR_DRY_AIR = ureg.Quantity(1.0002932, "")


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


@ureg.wraps(ureg.km ** -1,
            (ureg.nm,
             ureg.km ** -3,
             ureg.dimensionless, None, None),
            strict=False)
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
