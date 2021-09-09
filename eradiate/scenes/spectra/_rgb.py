from typing import MutableMapping, Optional

import attr
import pint
import pinttr

import eradiate

from ._core import Spectrum, spectrum_factory
from ... import unit_context_config as ucc
from ... import unit_context_kernel as uck
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError


@spectrum_factory.register(type_id="rgb")
@parse_docs
@attr.s
class RGBSpectrum(Spectrum):
    """
    Uniform spectrum (*i.e.* constant vs wavelength).
    """

    value = documented(
        attr.ib(default=[1.0, 1.0, 1.0]),
        doc="RGB spectrum values. If a list is passed and ``quantity`` is not "
        "``None``, it is automatically converted to appropriate configuration "
        "default units. If a :class:`~pint.Quantity` is passed and ``quantity`` "
        "is not ``None``, units will be checked for consistency.",
        type="float or :class:`~pint.Quantity`",
        default="[1.0, 1.0, 1.0] <quantity units>",
    )

    @value.validator
    def _value_validator(self, attribute, value):
        if type(value) not in [pint.Quantity, list] or len(value) != 3:
            raise TypeError("RGB spectrum values must have three components.")

        if self.quantity is not None and isinstance(value, pint.Quantity):
            expected_units = ucc.get(self.quantity)

            if not pinttr.util.units_compatible(expected_units, value.units):
                raise pinttr.exceptions.UnitsError(
                    value.units,
                    expected_units,
                    extra_msg=f"while validating {attribute.name}, got units "
                    f"'{value.units}' incompatible with quantity {self.quantity} "
                    f"(expected '{expected_units}')",
                )

    def __attrs_post_init__(self):
        if not eradiate.mode().has_flags("ANY_RGB"):
            raise UnsupportedModeError(
                msg="RGB spectra can only be used in rgb mode.", supported="rgb"
            )
        self.update()

    def update(self):
        if self.quantity is not None and self.value is not None:
            self.value = pinttr.converters.ensure_units(
                self.value, ucc.get(self.quantity)
            )

    def eval(self, spectral_ctx: SpectralContext = None) -> pint.Quantity:
        return self.value

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        kernel_units = uck.get(self.quantity)
        spectral_ctx = ctx.spectral_ctx if ctx is not None else None

        return {
            "spectrum": {
                "type": "rgb",
                "value": self.eval(spectral_ctx).m_as(kernel_units),
            }
        }
