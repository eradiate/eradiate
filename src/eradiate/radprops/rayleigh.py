"""Functions to compute Rayleigh scattering in air."""

import numpy as np
import numpy.typing as npt
import pint
from scipy.constants import physical_constants

from .. import converters
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
        self._king_factor = None
        self._wavelength = None

    @property
    def king_factor(self):
        if self._king_factor is None:
            self._load_dataset()
        return self._king_factor

    @property
    def wavelength(self):
        if self._wavelength is None:
            self._load_dataset()
        return self._wavelength

    def interp(self, w: npt.ArrayLike = np.array([0.550])):
        """
        Interpolate ``king_factor`` over the ``wavelength`` dimension.

        Parameters
        ----------
        w : array-like
            wavelengths (in micron) at wich to interpolate Bates' King factor.

        Returns
        -------
        ndarray
            Bates' King factor
        """
        left = self.king_factor[0]
        right = self.king_factor[-1]

        return np.interp(w, self.wavelength, self.king_factor, left=left, right=right)

    def _load_dataset(self):
        bates_data = converters.load_dataset("constant/optics/bates_1984.nc")
        self._king_factor = bates_data.f.values
        self._wavelength = bates_data.w.values


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
    # We are going to elevate `wavelength` to the power 4: if it is stored as a
    # 32-bit int, as can happen on Windows where the default integer type is
    # int32, calculus will be wrong due to integer overflow. To mitigate that
    # risk, we convert the wavelength to micron.
    # In addition, the Bates (1984) dataset is indexed by wavelengths in microns
    # as well, meaning that this conversion is anyway necessary.
    w = wavelength.to("micron")

    BATES_1984_DATA = _BATES_1984_DATA()
    king_factor = BATES_1984_DATA.interp(w.m_as("micron"))

    refractive_index = air_refractive_index(wavelength=w, number_density=number_density)
    if isinstance(w.magnitude, np.ndarray) and isinstance(
        number_density.magnitude, np.ndarray
    ):
        king_factor = king_factor[:, np.newaxis]
        w = w[:, np.newaxis]
        number_density = number_density[np.newaxis, :]

    result = (
        8.0
        * np.power(np.pi, 3)
        / (3.0 * np.power(w, 4))
        / number_density
        * np.square(np.square(refractive_index) - 1.0)
        * king_factor
    )
    return result.to("km^-1")


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


def depolarization_bates(wavelength: pint.Quantity = ureg.Quantity(550.0, "nm")):
    """
    Compute depolarization using Bates' King factor :cite:`Bates1984RayleighScatteringAir`.
    Only parametrized on wavelength.

    Parameters
    ----------
    wavelength : quantity
        Wavelength.

    Returns
    -------
    Quantity : scalar
        The depolarization factor parametrized on the wavelength. [dimensionless]
    """
    # The Bates (1984) dataset is indexed by wavelengths in microns
    # as well, meaning that this conversion is anyway necessary.
    w = wavelength.m_as("micron")

    BATES_1984_DATA = _BATES_1984_DATA()
    king_factor = BATES_1984_DATA.interp(w)

    depol = 6 * (king_factor - 1) / (7 * king_factor + 3)
    return np.atleast_1d(depol) * ureg.dimensionless


def depolarization_bodhaine(
    wavelength: pint.Quantity = ureg.Quantity(550.0, "nm"),
    x_CO2: pint.Quantity = ureg.Quantity(0.0004, "dimensionless"),
):
    """
    Compute depolarization using Bodhaine's King factor :cite:p:`Bodhaine1999RayleighOpticalDepth`.
    Parametrized over wavelength and CO2 concentration, other components
    are assumed to be at a fixed proportion. Valid at 273.15 K,
    1013.25 mb, for dry air.

    Parameters
    ----------
    wavelength : quantity
        Wavelength.

    x_CO2 : quantity
        Array of CO2 mole fraction in the atmosphere.

    Returns
    -------
    quantity : array of shape (N,)
        Array of depolarization factor for each level [dimensionless]
    """
    w_um = wavelength.m_as("um")

    # part per volume by percent
    C_CO2 = x_CO2.m_as("%")
    total = 78.084 + 20.946 + 0.934 + C_CO2

    # from Bates (1984), wavelength is in micron
    F_N2 = 1.034 + 3.17 * 1e-4 * 1 / w_um**2
    F_O2 = 1.096 + 1.385 * 1e-3 * 1 / w_um**2 + 1.448 * 1e-4 * 1 / w_um**4
    F_air = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 * 1.00 + C_CO2 * 1.15) / total

    # calculate the depolarization
    return 6 * (F_air - 1) / (7 * F_air + 3) * ureg.dimensionless
