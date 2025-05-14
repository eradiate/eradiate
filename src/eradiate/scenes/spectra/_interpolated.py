from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

from ._core import Spectrum
from ... import converters, validators
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...units import PhysicalQuantity, to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg

_trapezoid = np.trapezoid if int(np.__version__.split(".")[0]) >= 2 else np.trapz


@define(eq=False, slots=False, init=False)
class InterpolatedSpectrum(Spectrum):
    """
    Linearly interpolated spectrum [``interpolated``].

    .. admonition:: Class method constructors

       .. autosummary::

          from_dataarray

    Warnings
    --------
    In all modes, queries outside the spectral domain covered by the input data
    return 0.

    Notes
    -----
    * Interpolation uses :func:`numpy.interp`. In both ``mono_*`` and ``ckd_*``
      modes, the spectrum is evaluated at the spectral context wavelength.
    * Integration performed with the :meth:`.integral` method uses a trapezoid
      rule.
    """

    wavelengths: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            converter=[
                np.atleast_1d,
                pinttr.converters.to_units(ucc.deferred("wavelength")),
            ],
            kw_only=True,
        ),
        doc="Wavelengths defining the interpolation grid. Values must be "
        "monotonically increasing.",
        type="quantity",
    )

    @wavelengths.validator
    def _wavelengths_validator(self, attribute, value):
        # Wavelengths must be monotonically increasing
        if not np.all(np.diff(value) > 0):
            raise ValueError("wavelengths must be monotonically increasing")

        # Check values
        if np.any(np.isnan(value)):
            raise ValueError("Detected NaN in 'wavelengths'")

    values: pint.Quantity = documented(
        attrs.field(
            converter=converters.on_quantity(np.atleast_1d),
            kw_only=True,
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
    def _values_validator(self, attribute, value):
        if self.quantity is not None:
            if not isinstance(self.values, pint.Quantity):
                raise ValueError(
                    f"while validating '{attribute.name}': expected a Pint "
                    "quantity compatible with quantity field value "
                    f"'{self.quantity}', got a unitless value instead"
                )

            expected_units = ucc.get(self.quantity)
            if not pinttr.util.units_compatible(expected_units, value.units):
                raise pinttr.exceptions.UnitsError(
                    value.units,
                    expected_units,
                    extra_msg=f"while validating '{attribute.name}', got units "
                    f"'{value.units}' incompatible with quantity {self.quantity} "
                    f"(expected '{expected_units}')",
                )

    @values.validator
    @wavelengths.validator
    def _values_wavelengths_validator(self, attribute, value):
        # Check that attribute is an array
        validators.on_quantity(attrs.validators.instance_of(np.ndarray))(
            self, attribute, value
        )

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

        # Check values
        if np.any(np.isnan(value)):
            raise ValueError("Detected NaN in 'values'")

    def __init__(
        self,
        id: str | None = None,
        quantity: PhysicalQuantity | str | None = None,
        *,
        wavelengths: np.typing.ArrayLike,
        values: np.typing.ArrayLike,
    ):
        # If a quantity is set and a unitless value is passed, it is
        # automatically applied appropriate units
        if quantity is not None and not isinstance(values, pint.Quantity):
            values = pinttr.util.ensure_units(values, ucc.get(quantity))

        self.__attrs_init__(
            id=id, quantity=quantity, wavelengths=wavelengths, values=values
        )

    def update(self):
        # Sort by increasing wavelength (required by numpy.interp in eval_mono)
        idx = np.argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[idx]
        self.values = self.values[idx]

    @classmethod
    def from_dataarray(
        cls,
        id: str | None = None,
        quantity: str | PhysicalQuantity | None = None,
        *,
        dataarray: xr.DataArray,
    ) -> InterpolatedSpectrum:
        """
        Construct an interpolated spectrum from an xarray data array.

        Parameters
        ----------
        id : str, optional
            Optional object identifier.

        quantity : str or .PhysicalQuantity, optional
            If set, quantity represented by the spectrum. This parameter and
            spectrum units must be consistent. This parameter takes precedence
            over the ``quantity`` field of the data array.

        dataarray : DataArray
            An :class:`xarray.DataArray` instance complying to the spectrum data
            array format (see *Notes*).

        Notes
        -----

        * Expected data format:

          **Coordinates (\\* means also dimension)**

          * ``*w`` (float): wavelength (units specified as a ``units`` attribute).

          **Metadata**

          * ``quantity`` (str): physical quantity which the data describes (see
            :meth:`.PhysicalQuantity.spectrum` for allowed values), optional.
          * ``units`` (str): units of spectrum values (must be consistent with
            ``quantity``).
        """
        kwargs = {}

        if id is not None:
            kwargs[id] = id

        if quantity is None:
            try:
                kwargs["quantity"] = dataarray.attrs["quantity"]
            except KeyError:
                pass
        else:
            kwargs["quantity"] = quantity

        values = to_quantity(dataarray)
        wavelengths = to_quantity(dataarray.w)

        return cls(**kwargs, values=values, wavelengths=wavelengths)

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return np.interp(w, self.wavelengths, self.values, left=0.0, right=0.0)

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_mono(w=w)

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        # Inherit docstring

        # Collect spectral coordinates
        wavelength_units = self.wavelengths.units
        s_w = self.wavelengths.m
        s_wmin = s_w.min()
        s_wmax = s_w.max()

        # Select all spectral mesh vertices between wmin and wmax, as well as
        # the bounds themselves
        wmin = wmin.m_as(wavelength_units)
        wmax = wmax.m_as(wavelength_units)
        w = [wmin, *s_w[np.where(np.logical_and(wmin < s_w, s_w < wmax))[0]], wmax]

        # Make explicit the fact that the function underlying this spectrum has
        # a finite support by padding the s_wmin and s_wmax values with a very
        # small margin
        eps = 1e-12  # nm

        try:
            w.insert(w.index(s_wmin), s_wmin - eps)
        except ValueError:
            pass

        try:
            w.insert(w.index(s_wmax) + 1, s_wmax + eps)
        except ValueError:
            pass

        # Evaluate spectrum at wavelengths
        interp = self.eval_mono(w * wavelength_units)

        # Compute integral
        integral = _trapezoid(interp, w)

        # Apply units
        return integral * ureg.dimensionless * wavelength_units

    @property
    def template(self) -> dict:
        # Inherit docstring

        return {
            "type": "uniform",
            "value": DictParameter(
                func=lambda ctx: float(self.eval(ctx.si).m_as(uck.get(self.quantity)))
            ),
        }

    @property
    def params(self) -> dict:
        # Inherit docstring

        return {
            "value": SceneParameter(
                func=lambda ctx: float(self.eval(ctx.si).m_as(uck.get(self.quantity))),
                flags=KernelSceneParameterFlags.SPECTRAL,
            )
        }
