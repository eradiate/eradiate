from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr

from ._core import Spectrum
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck


@parse_docs
@attrs.define
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum [``uniform``] (*i.e.* constant vs wavelength).
    """

    value: pint.Quantity = documented(
        attrs.field(
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

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_mono(w)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        kernel_units = uck.get(self.quantity)
        value = float(self.eval(ctx.spectral_index).m_as(kernel_units))
        return KernelDict({"spectrum": {"type": "uniform", "value": value}})

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        wmin = pinttr.util.ensure_units(wmin, ucc.get("wavelength"))
        wmax = pinttr.util.ensure_units(wmax, ucc.get("wavelength"))
        return self.value * (wmax - wmin)
