from __future__ import annotations

import attr
import numpy as np
import pint
import pinttr

from ._core import Spectrum, spectrum_factory
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@spectrum_factory.register(type_id="uniform")
@parse_docs
@attr.s
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum (*i.e.* constant vs wavelength).
    """

    value: pint.Quantity = documented(
        attr.ib(default=1.0, repr=lambda x: f"{x:~P}"),
        doc="Uniform spectrum value. If a float is passed and ``quantity`` is not "
        "``None``, it is automatically converted to appropriate configuration "
        "default units. If a :class:`~pint.Quantity` is passed and ``quantity`` "
        "is not ``None``, units will be checked for consistency.",
        type="float or :class:`~pint.Quantity`",
        default="1.0 <quantity units>",
    )

    @value.validator
    def _value_validator(self, attribute, value):
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
        self.update()

    def update(self):
        if self.quantity is not None and self.value is not None:
            self.value = pinttr.converters.ensure_units(
                self.value, ucc.get(self.quantity)
            )

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        if isinstance(self.value, pint.Quantity):
            return np.full_like(w, self.value.m) * self.value.units
        else:
            return np.full_like(w, self.value) * ureg.dimensionless

    def eval_rgb(self) -> pint.Quantity:
        if isinstance(self.value, pint.Quantity):
            return np.full_like(3, self.value.m) * self.value.units
        else:
            return np.full_like(3, self.value) * ureg.dimensionless

    def eval_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        if isinstance(self.value, pint.Quantity):
            return np.full((len(bindexes),), self.value.m) * self.value.units
        else:
            return np.full((len(bindexes),), self.value) * ureg.dimensionless

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        kernel_units = uck.get(self.quantity)
        spectral_ctx = ctx.spectral_ctx

        return KernelDict(
            {
                "spectrum": {
                    "type": "uniform",
                    "value": float(self.eval(spectral_ctx).m_as(kernel_units)),
                }
            }
        )
