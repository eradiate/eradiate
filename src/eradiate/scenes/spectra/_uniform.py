from __future__ import annotations

import attr
import numpy as np
import pint
import pinttr

from ._core import Spectrum
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck


@parse_docs
@attr.s
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum [``uniform``] (*i.e.* constant vs wavelength).
    """

    value: pint.Quantity = documented(
        attr.ib(
            converter=lambda x: float(x) if isinstance(x, int) else x,
            repr=lambda x: f"{x:~P}",
            kw_only=True,
        ),
        doc="Uniform spectrum value. If a float is passed, it is automatically "
        "converted to appropriate configuration default units. Integer values "
        "are also converted to float.",
        type=":class:`~pint.Quantity`",
        init_type="float or :class:`~pint.Quantity` or int",
    )

    @value.validator
    def _value_validator(self, attribute, value):
        if isinstance(value, pint.Quantity):
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
        self.value = pinttr.util.ensure_units(self.value, ucc.get(self.quantity))

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return np.full_like(w, self.value.m) * self.value.units

    def eval_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        return np.full((len(bindexes),), self.value.m) * self.value.units

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        wmin = pinttr.util.ensure_units(wmin, ucc.get("wavelength"))
        wmax = pinttr.util.ensure_units(wmax, ucc.get("wavelength"))
        return self.value * (wmax - wmin)

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
