"""Functions to compute Rayleigh scattering in air."""

import numpy as np
import pint
from scipy.constants import physical_constants

from .. import data
from ..units import unit_registry as ureg
from ..util.misc import Singleton

# Physical constants
#: Loschmidt constant [km^-3].
_LOSCHMIDT = ureg.Quantity(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2]
).to("km^-3")

# Air number density at 101325 Pa and 288.15 K
_STANDARD_AIR_NUMBER_DENSITY = _LOSCHMIDT * (273.15 / 288.15)


# Bates (1984) King correction factor data
class _BATES_1984_DATA(metaclass=Singleton):
    # Lazy loader for Bates (1984) King correction data
    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = data.load_dataset("spectra/optics/bates_1984.nc")
        return self._data


def compute_sigma_s_air(
    wavelength: pint.Quantity = ureg.Quantity(550.0, "nm"),
    number_density: pint.Quantity = _STANDARD_AIR_NUMBER_DENSITY,
) -> pint.Quantity:
    r"""
    Compute the Rayleigh scattering coefficient of air.

    When default values are used, this provides the Rayleigh scattering
    coefficient for air at 550 nm in standard temperature and pressure
    conditions.

    The scattering coefficient is computed by considering the air as a pure
    gas with associated effective optical properties (refractive index,
    King factor) and according to the expression provided by
    :cite:`Eberhard2010CorrectEquationsCommon` (eq. 60):

    .. math::

       k_{\mathrm s \, \lambda} (n) = \frac{8 \pi^3}{3 \lambda^4} \frac{1}{n}
          \left( \eta_{\lambda}^2(n) - 1 \right)^2 F_{\lambda}

    where
    :math:`\lambda` is the wavelength (subscript indicates spectral dependence),
    :math:`n` is the air number density,
    :math:`\eta` is the air refractive index and
    :math:`F` is the air King factor.

    The King correction factor is computed by linearly interpolating the data
    from :cite:`Bates1984RayleighScatteringAir`.

    Parameters
    ----------
    wavelength : quantity
        Wavelength [nm].

    number_density : quantity
        Number density of the scattering particles [km^-3].

    Returns
    -------
    quantity
        Scattering coefficient.
    """
    BATES_1984_DATA = _BATES_1984_DATA().data
    f_left = BATES_1984_DATA.f.values[0]
    f_right = BATES_1984_DATA.f.values[-1]
    king_factor = BATES_1984_DATA.f.interp(
        w=wavelength.m_as("micrometer"),
        kwargs=dict(fill_value=(f_left, f_right)),
    ).values

    refractive_index = air_refractive_index(
        wavelength=wavelength, number_density=number_density
    )
    if isinstance(wavelength.magnitude, np.ndarray) and isinstance(
        number_density.magnitude, np.ndarray
    ):
        king_factor = king_factor[:, np.newaxis]
        wavelength = wavelength[:, np.newaxis]
        number_density = number_density[np.newaxis, :]

    return (
        8.0
        * np.power(np.pi, 3)
        / (3.0 * np.power(wavelength, 4))
        / number_density
        * np.square(np.square(refractive_index) - 1.0)
        * king_factor
    ).to("km^-1")


def air_refractive_index(
    wavelength: pint.Quantity = ureg.Quantity(550.0, "nm"),
    number_density: pint.Quantity = _STANDARD_AIR_NUMBER_DENSITY,
) -> np.ndarray:
    """
    Computes the air refractive index.

    The wavelength dependence of the refractive index is computed using equation
    2 from :cite:`Peck1972DispersionAir`. This formula is a fit of
    measurements of the air refractive index in the range of wavelength from
    :math:`\\lambda = 240` nm to :math:`1690` nm.
    The number density dependence is computed using a simple proportionality
    rule.

    Parameters
    ----------
    wavelength : quantity
        Wavelength.

    number_density : quantity
        Number density.

        Default: Air number density at 101325 Pa and 288.15 K.

    Returns
    -------
    float or ndarray
        Air refractive index value(s).
    """

    # wavenumber in inverse micrometer
    sigma = 1 / wavelength.m_as("micrometer")
    sigma2 = np.square(sigma)

    # refractivity in parts per 1e8
    x = (5791817.0 / (238.0183 - sigma2)) + 167909.0 / (57.362 - sigma2)

    if isinstance(x, np.ndarray) and isinstance(number_density.magnitude, np.ndarray):
        x = x[:, np.newaxis]
        number_density = number_density[np.newaxis, :]

    # number density scaling
    x_scaled = x * (number_density / _STANDARD_AIR_NUMBER_DENSITY).m_as("dimensionless")

    # refractive index
    index = 1 + x_scaled * 1e-8

    return index
