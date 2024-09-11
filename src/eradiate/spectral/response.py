from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pint
import pinttrs
import scipy.integrate as spi
import xarray as xr
from pinttrs.util import ensure_units

from .. import converters, data, validators
from ..attrs import define, documented
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import summary_repr

# TODO:
#  * Propagate to measure and other intermediate components
#  * Add plotting facilities
#  * Documentation


# ------------------------------------------------------------------------------
#                             Class implementations
# ------------------------------------------------------------------------------


@define
class SpectralResponseFunction(ABC):
    """
    An interface defining the spectral response function of an instrument.
    """

    @abstractmethod
    def eval(self, w: npt.ArrayLike) -> pint.Quantity:
        pass


@define
class UniformSRF(SpectralResponseFunction):
    """
    A spectral response function uniform on a preset spectral interval.
    """

    wmin = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            factory=lambda: 300.0 * ureg.nm,
            validator=validators.is_positive,
        )
    )

    wmax = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            factory=lambda: 2500.0 * ureg.nm,
            validator=validators.is_positive,
        )
    )

    value = documented(
        pinttrs.field(
            units=ucc.deferred("dimensionless"),
            factory=lambda: 1.0,
            validator=validators.is_positive,
        )
    )

    def eval(self, w: npt.ArrayLike) -> pint.Quantity:
        # Inherit docstring

        w_units = ucc.get("wavelength")
        w_m = pinttrs.util.ensure_units(w, w_units).m_as(w_units)

        return (
            np.where(
                (w_m >= self.wmin.m_as(w_units)) & (w_m <= self.wmax.m_as(w_units)),
                self.value.m,
                np.zeros_like(w),
            )
            * self.value.u
        )


@define
class DeltaSRF(SpectralResponseFunction):
    """
    A spectral response function consisting of multiple Dirac delta distributions.
    """

    wavelengths: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            converter=[
                converters.on_quantity(np.atleast_1d),
                converters.on_quantity(lambda x: x.astype(np.float64)),
                pinttrs.converters.to_units(ucc.deferred("wavelength")),
            ],
            validator=validators.all_strictly_positive,
            repr=summary_repr,
        ),
        doc="An array of wavelengths specifying the translation wavelength of each "
        "Dirac delta. Wavelength values are positive and unique. "
        "When a single value is provided, it is converted to a 1-element array. "
        "Wavelength are deduplicated and sorted by ascending values. "
        'Unit-enabled field (default: ``ucc["wavelength"]``).',
        type="quantity",
        init_type="array-like or quantity",
    )

    def eval(self, w: npt.ArrayLike) -> pint.Quantity:
        # Inherit docstring

        return np.zeros_like(w) * ureg.dimensionless


@define
class BandSRF(SpectralResponseFunction):
    """
    The spectral response function for a single band of an instrument.
    """

    wavelengths: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            converter=[
                converters.on_quantity(np.atleast_1d),
                pinttrs.converters.to_units(ucc.deferred("wavelength")),
            ],
            validator=validators.is_sorted(strict=True),
            repr=summary_repr,
        ),
        doc="Wavelengths defining the interpolation grid. Values must be "
        "monotonically increasing.",
        type="quantity",
    )

    values: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("dimensionless"),
            converter=[
                converters.on_quantity(np.atleast_1d),
                pinttrs.converters.to_units(ucc.deferred("dimensionless")),
            ],
        ),
        doc="Spectrum values. If a quantity is passed, units are "
        "checked for consistency. If a unitless array is passed, it is "
        "automatically converted to appropriate default configuration units, "
        "depending on the value of the ``quantity`` field. If no quantity is "
        "specified, this field can be a unitless value.",
        type="quantity",
        init_type="array-like",
    )

    @values.validator
    @wavelengths.validator
    def _values_wavelengths_validator(self, attribute, value):
        # Check size
        if value.ndim > 1:
            raise ValueError(
                f"while validating '{attribute.name}': '{attribute.name}' must "
                "be a 1D array"
            )

        if len(value) < 2:
            raise ValueError(
                f"while validating '{attribute.name}': '{attribute.name}' must "
                f"have length >= 2"
            )

        if self.wavelengths.shape != self.values.shape:
            raise ValueError(
                f"while validating '{attribute.name}': 'wavelengths' and 'values' "
                f"must have the same shape, got {self.wavelengths.shape} and "
                f"{self.values.shape}"
            )

    @values.validator
    def _values_validator(self, attribute, value):
        # Check for leading and trailing zeros
        if not np.all(value.m[[0, -1]] == np.array([0.0, 0.0])):
            warnings.warn(
                "Initializing a BandSRF instance without a leading and trailing "
                "zero is not recommended."
            )

    @classmethod
    def from_dataarray(cls, da: xr.DataArray):
        return cls(wavelengths=to_quantity(da.w), values=da.values)

    @classmethod
    def from_id(cls, id: str):
        ds = data.load_dataset(f"spectra/srf/{id}.nc")
        return cls.from_dataarray(ds.srf)

    def support(self) -> pint.Quantity:
        """
        Return the interval in which the SRF is nonzero.

        Note
        ----
        This method does not actually compute the support: it assumes that any
        leading and trailing zeros are already removed.
        """
        wmin_m = self.wavelengths.m.min()
        wmax_m = self.wavelengths.m.max()
        return (wmin_m, wmax_m) * self.wavelengths.u

    def eval(self, w: npt.ArrayLike) -> pint.Quantity:
        # Inherit docstring

        w_units = self.wavelengths.u
        w_m = pinttrs.util.ensure_units(w, ucc.get("wavelength")).m_as(w_units)
        return (
            np.interp(w_m, self.wavelengths.m, self.values.m, left=0.0, right=0.0)
            * self.values.u
        )

    def integrate(
        self, wmin: float | pint.Quantity, wmax: float | pint.Quantity
    ) -> pint.Quantity:
        """
        Return the integral of the SRF on the specified interval, using the
        trapezoid rule.

        Parameters
        ----------
        wmin, wmax : float or quantity
            Lower and upper bounds of the integration domain. Floats are
            interpreted as being specified in default configuration units.

        Returns
        -------
        integral : quantity
            Integral as a scalar quantity.
        """
        wmin = ensure_units(wmin, ucc.get("wavelength"))
        wmax = ensure_units(wmax, ucc.get("wavelength"))

        # Assemble spectral integration mesh
        w_u = self.wavelengths.u
        w_m = np.unique(
            np.concatenate(
                (
                    np.atleast_1d(wmin.m_as(w_u)),
                    self.wavelengths.m[
                        (self.wavelengths >= wmin) & (self.wavelengths <= wmax)
                    ],
                    np.atleast_1d(wmax.m_as(w_u)),
                )
            )
        )

        # Evaluate SRF at mesh nodes
        values_m = self.eval(w_m * w_u).m

        # Compute integral
        return spi.trapezoid(values_m, w_m) * w_u

    def integrate_cumulative(self, w: npt.ArrayLike) -> pint.Quantity:
        """
        Return the cumulative integral of the SRF on the specified mesh, using
        the trapezoid rule.

        Parameters
        ----------
        w : array-like
            Nodes of the spectral integration mesh. If a dimensionless array is
            passed, it is interpreted as being specified in default
            configuration units.

        Returns
        -------
        integral : quantity
            Cumulative integral as an array of shape (N-1,), where
            ``wavelength`` has shape (N,).
        """
        w_u = ucc.get("wavelength")
        w_m = ensure_units(w, w_u).m_as(w_u)

        # Evaluate SRF at mesh nodes
        values_m = self.eval(w).m

        # Compute integral
        return spi.cumulative_trapezoid(values_m, w_m) * w_u