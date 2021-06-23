from typing import MutableMapping, Optional

import attr
import numpy as np
import pint
import pinttr

import eradiate

from ._core import Spectrum, SpectrumFactory
from ... import converters, validators
from ..._attrs import documented, parse_docs
from ..._util import ensure_array
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck


@SpectrumFactory.register("interpolated")
@parse_docs
@attr.s
class InterpolatedSpectrum(Spectrum):
    """
    Linearly interpolated spectrum. Interpolation uses :func:`numpy.interp`.
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

    def eval(self, spectral_ctx: SpectralContext = None) -> pint.Quantity:
        if spectral_ctx is None:
            raise ValueError("spectral_ctx must not be None")

        if eradiate.mode().is_monochromatic():
            return np.interp(spectral_ctx.wavelength, self.wavelengths, self.values)
        else:
            raise UnsupportedModeError(supported="monochromatic")

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        kernel_units = uck.get(self.quantity)
        spectral_ctx = ctx.spectral_ctx if ctx is not None else None

        if eradiate.mode().is_monochromatic():
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": self.eval(spectral_ctx).m_as(kernel_units),
                }
            }

        else:
            raise UnsupportedModeError(supported="monochromatic")
