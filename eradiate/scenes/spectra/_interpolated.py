from __future__ import annotations

import attr
import numpy as np
import pint
import pinttr

import eradiate

from ._core import Spectrum, spectrum_factory
from ..core import KernelDict
from ... import converters, validators
from ..._mode import ModeFlags
from ..._util import ensure_array
from ...attrs import documented, parse_docs
from ...ckd import Bin
from ...contexts import KernelDictContext
from ...exceptions import UnsupportedModeError
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@spectrum_factory.register(type_id="interpolated")
@parse_docs
@attr.s
class InterpolatedSpectrum(Spectrum):
    """
    Linearly interpolated spectrum. Interpolation uses :func:`numpy.interp`.

    Evaluation is as follows:

    * in ``mono_*`` modes, the spectrum is evaluated at the spectral context
      wavelength;
    * in ``ckd_*`` modes, the spectrum is evaluated as the average value over
      the spectral context bin (the integral is computed using a trapezoid
      rule).
    """

    wavelengths: pint.Quantity = documented(
        pinttr.ib(
            default=None,
            units=ucc.deferred("wavelength"),
        ),
        doc="Wavelengths defining the interpolation grid.",
        type=":class:`pint.Quantity`",
    )

    values: pint.Quantity = documented(
        attr.ib(default=None, converter=converters.on_quantity(ensure_array)),
        doc="Uniform spectrum value. If a float is passed and ``quantity`` is not "
        "``None``, it is automatically converted to appropriate configuration "
        "default units. If a :class:`~pint.Quantity` is passed and ``quantity`` "
        "is not ``None``, units will be checked for consistency.",
        type="float or :class:`~pint.Quantity`",
    )

    @values.validator
    def _values_validator(self, attribute, value):
        if self.quantity is not None and isinstance(value, pint.Quantity):
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
        validators.on_quantity(attr.validators.instance_of(np.ndarray))(
            self, attribute, value
        )

        # Check size
        if value.ndim > 1:
            f"while validating '{attribute.name}': '{attribute.name}' must be a 1D array"

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

    def __attrs_post_init__(self):
        self.update()

    def update(self):
        # Apply appropriate units to values field
        if self.quantity is not None and self.values is not None:
            self.values = pinttr.converters.ensure_units(
                self.values, ucc.get(self.quantity)
            )

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return np.interp(w, self.wavelengths, self.values, left=0.0, right=0.0)

    def eval_ckd(self, *bins: Bin) -> pint.Quantity:
        # Spectrum is averaged over spectral bin

        result = np.zeros((len(bins),))
        wavelength_units = ucc.get("wavelength")
        quantity_units = (
            self.values.units if hasattr(self.values, "units") else ureg.dimensionless
        )

        for i_bin, bin in enumerate(bins):
            wmin_m = bin.wmin.m_as(wavelength_units)
            wmax_m = bin.wmax.m_as(wavelength_units)

            # -- Collect relevant spectral coordinate values
            w_m = self.wavelengths.m_as(wavelength_units)
            w = (
                np.hstack(
                    (
                        [wmin_m],
                        w_m[np.where(np.logical_and(wmin_m < w_m, w_m < wmax_m))[0]],
                        [wmax_m],
                    )
                )
                * wavelength_units
            )

            # -- Evaluate spectrum at wavelengths
            interp = self.eval_mono(w)

            # -- Average spectrum on bin extent
            integral = np.trapz(interp, w)
            result[i_bin] = (integral / bin.width).m_as(quantity_units)

        return result * quantity_units

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
            return KernelDict(
                {
                    "spectrum": {
                        "type": "uniform",
                        "value": float(
                            self.eval(ctx.spectral_ctx).m_as(uck.get(self.quantity))
                        ),
                    }
                }
            )

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))
