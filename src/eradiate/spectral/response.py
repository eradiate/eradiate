from __future__ import annotations

import os
import typing as t
import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pint
import pinttrs
import scipy.integrate as spi
import xarray as xr
from pinttrs.util import ensure_units

from .. import converters, validators
from ..attrs import define, documented
from ..data import fresolver
from ..exceptions import DataError
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import summary_repr

# ------------------------------------------------------------------------------
#                             Class implementations
# ------------------------------------------------------------------------------


@define
class SpectralResponseFunction(ABC):
    """
    An interface defining the spectral response function of an instrument.
    """

    @staticmethod
    def convert(value) -> t.Any:
        """
        Converter for the ``Measure.srf`` field.

        Notes
        -----
        Supported conversion protocols:

        * :class:`dict`: Dispatch to subclass based on 'type' entry, then pass
          dictionary to constructor as keyword arguments.
        * Dataset, DataArray: Call :meth:`.BandSRF.from_dataarray`.
        * Path-like: Attempt loading a dataset from the hard drive, then call
          :meth:`.BandSRF.from_dataarray`.
        * :class:`str`: Perform a NetCDF file lookup in the SRF database and load
          it.

        Anything else will pass through this converter without modification.
        """
        if isinstance(value, dict):
            d = value.copy()
            try:
                type_id = d.pop("type")
            except KeyError as e:
                raise ValueError(
                    "missing 'type' key in SRF specification dictionary"
                ) from e

            dispatch_table = {
                "uniform": UniformSRF,
                "delta": DeltaSRF,
                "multi_delta": DeltaSRF,
                "band": BandSRF,
            }

            try:
                cls = dispatch_table[type_id]
            except KeyError as e:
                raise ValueError(f"unknown SRF type '{type_id}'") from e

            return cls(**d)

        if isinstance(value, xr.Dataset):
            return BandSRF.from_dataarray(value.srf)

        if isinstance(value, xr.DataArray):
            return BandSRF.from_dataarray(value)

        if isinstance(value, (str, os.PathLike)):
            try:
                ds = xr.load_dataset(value)
                return BandSRF.from_dataarray(ds.srf)
            except (FileNotFoundError, ValueError):
                pass

        if isinstance(value, str):
            try:
                return BandSRF.from_id(value)
            except DataError:
                pass

        return value

    @abstractmethod
    def eval(self, w: npt.ArrayLike) -> pint.Quantity:
        """
        Evaluate the spectral response function for one or several wavelengths.
        Evaluation is vectorized.

        Parameters
        ----------
        w : array-like
            One or several wavelengths at which the SRF is evaluated.

        Returns
        -------
        quantity
            The returned value as the same shape as ``w``.
        """
        pass


@define
class UniformSRF(SpectralResponseFunction):
    r"""
    A spectral response function uniform on a preset spectral interval.

    It represents a function defined by:

    .. math::

       f(w) = \left\{
       \begin{array}{r l}
            \text{value} & \text{if} \ w \in [w_\mathrm{min}, w_\mathrm{max}] \\
            0 & \text{otherwise}
        \end{array}
        \right.
    """

    wmin = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            factory=lambda: 300.0 * ureg.nm,
            validator=validators.is_positive,
        ),
        doc="Lower bound of the interval.",
        type="quantity",
        init_type="quantity or float",
        default="300 nm",
    )

    wmax = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            factory=lambda: 2500.0 * ureg.nm,
            validator=validators.is_positive,
        ),
        doc="Upper bound of the interval.",
        type="quantity",
        init_type="quantity or float",
        default="2500 nm",
    )

    value = documented(
        pinttrs.field(
            units=ucc.deferred("dimensionless"),
            factory=lambda: 1.0,
            validator=validators.is_positive,
        ),
        doc="The value to which the phase function evaluates in its interval.",
        type="quantity",
        init_type="quantity or float",
        default="1.0",
    )

    def plot(self, ax, alpha=0.5):
        w_u = ucc.get("wavelength")
        x = [self.wmin.m_as(w_u), self.wmax.m_as(w_u)]
        y = [self.value.m, self.value.m]

        ax.fill_between(x, y, alpha=alpha)
        ax.plot(x, y, marker=".")
        ax.vlines(x, 0, 1)

        return ax

    def _repr_html_(self):
        import base64
        import io

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

        self.plot(ax)
        w_u = ucc.get("wavelength")
        ax.set_xlabel(f"Wavelength [{w_u:~P}]")

        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        plt.close(fig)
        img.seek(0)

        return (
            "<img "
            f'src="data:image/png;base64, {base64.b64encode(img.getvalue()).decode("utf-8")}" '
            "/>"
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

    This SRF conventionally always evaluates to 0.
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

    def plot(self, ax):
        ax.vlines(self.wavelengths.m, 0, 1)
        return ax

    def _repr_html_(self):
        import base64
        import io

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1, 1, figsize=(6, 1))
        self.plot(ax)
        ax.set_xlabel(f"Wavelength [{self.wavelengths.u:~P}]")
        sns.despine(left=True)
        ax.axes.get_yaxis().set_visible(False)

        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        plt.close(fig)
        img.seek(0)

        return (
            "<img "
            f'src="data:image/png;base64, {base64.b64encode(img.getvalue()).decode("utf-8")}" '
            "/>"
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
        fname = fresolver.resolve(f"srf/{id}.nc")
        try:
            ds = xr.load_dataset(fname)
        except FileNotFoundError as e:
            raise DataError(f"could not load SRF with identifier '{id}'") from e

        return cls.from_dataarray(ds["srf"])

    def plot(self, ax, alpha=0.5, lw=1):
        w_u = ucc.get("wavelength")
        x = self.wavelengths.m_as(w_u)
        y = self.values.m
        if alpha:
            ax.fill_between(x, y, alpha=alpha)
        ax.plot(x, y, lw=lw, marker=".")
        return ax

    def _repr_html_(self):
        import base64
        import io

        import matplotlib.pyplot as plt

        w_u = ucc.get("wavelength")
        fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

        self.plot(ax)
        ax.set_xlabel(f"Wavelength [{w_u:~P}]")

        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        plt.close(fig)
        img.seek(0)

        return (
            "<img "
            f'src="data:image/png;base64, {base64.b64encode(img.getvalue()).decode("utf-8")}" '
            "/>"
        )

    def support(self) -> pint.Quantity:
        """
        Return the interval in which the SRF is nonzero.

        Returns
        -------
        quantity

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


@ureg.wraps(ret=None, args=("nm", "nm", None, "nm", "nm", None, None), strict=False)
def make_gaussian(
    wl_center: pint.Quantity,
    fwhm: pint.Quantity,
    cutoff: float = 3.0,
    wl: pint.Quantity | None = None,
    wl_res: pint.Quantity | float = 1.0,
    pad: bool = False,
    normalize: bool = True,
) -> xr.Dataset:
    """
    Generate a Gaussian spectral response function dataset from central
    wavelength and full width at half maximum values.

    Parameters
    ----------
    wl_center : quantity or float
        Central wavelength of the Gaussian distribution.
        If passed as a float, the value is interpreted as being given in nm.

    fwhm : quantity or float
        Full width at half maximum of the Gaussian distribution.
        If passed as a float, the value is interpreted as being given in nm.

    cutoff : float, default: 3.0
        Cut-off, in multiples of the standard deviation σ.

    wl : quantity or array-like, optional
        Mesh used to evaluate the discretized distribution. If unset, a regular
        mesh with spacing given by ``wl_res`` is used.
        If passed as an array, the value is interpreted as being given in nm.

    wl_res : quantity or float, optional
        Resolution of the automatic spectral mesh if relevant.
        If passed as a float, the value is interpreted as being given in nm.

    pad : bool, default: False
        If True, pad SRF data with leading and trailing zeros.

    normalize : bool, default: True
        If ``True``, the generated SRF data is normalized to have a maximum
        equal to 1.

    Returns
    -------
    Dataset
        A dataset compliant with the Eradiate SRF format. The uncertainty
        variable is set to NaN.
    """
    # Generate default mesh if necessary
    if wl is None:
        wl = np.arange(0.0, 5001.0, wl_res)

    # Infer standard deviation
    sigma = 0.5 * fwhm / np.sqrt(2.0 * np.log(2.0))

    # Build baseline distribution
    values = np.exp(-0.5 * np.power((wl - wl_center) / sigma, 2)) / (sigma * np.sqrt(2))

    # Prepare cutoff selection
    wl_min = wl_center - cutoff * sigma
    wl_max = wl_center + cutoff * sigma
    wl_mask = (wl >= wl_min) & (wl <= wl_max)

    if pad:  # Apply zero-padding if requested
        i_min = np.argwhere(wl_mask).min() - 1
        i_max = np.argwhere(wl_mask).max() + 1
        wl_mask[[i_min, i_max]] = True
        values_result = values[wl_mask]
        values_result[[0, -1]] = 0.0
    else:  # Otherwise just select
        values_result = values[wl_mask]

    # Build output dataset
    if normalize:
        values_result /= values_result.max()
    wl_result = wl[wl_mask]

    result = xr.Dataset(
        data_vars={
            "srf": (
                ["w"],
                values_result,
                {
                    "standard_name": "spectral_response_function",
                    "long_name": "spectral response function",
                    "units": "dimensionless",
                },
            ),
        },
        coords={
            "w": (
                ["w"],
                wl_result,
                {
                    "standard_name": "radiation_wavelength",
                    "long_name": "wavelength",
                    "units": "nm",
                },
            )
        },
    )
    result["srf_u"] = xr.full_like(result.srf, np.nan)
    result.srf_u.attrs = {
        "standard_name": "spectral_response_function_uncertainty",
        "long_name": "spectral response function uncertainty",
        "units": "dimensionless",
    }

    return result
