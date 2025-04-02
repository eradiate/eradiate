from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr

from ._core import Spectrum
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...units import PhysicalQuantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@define(eq=False, slots=False, init=False)
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum [``uniform``] (*i.e.* constant vs wavelength).
    """

    value: pint.Quantity = documented(
        attrs.field(
            converter=lambda x: float(x) if isinstance(x, int) else x,
            repr=lambda x: f"{x:~P}" if isinstance(x, pint.Quantity) else str(x),
            kw_only=True,
        ),
        doc="Uniform spectrum value. If a quantity is passed, units are "
        "checked for consistency. If a unitless value is passed, it is "
        "automatically converted to appropriate configuration default units, "
        "depending on the value of the ``quantity`` field. Integer values "
        "are converted to float. If no quantity is specified, this field must be"
        "a unitless value.",
        type="quantity",
        init_type="quantity or float or int",
    )

    @value.validator
    def _value_validator(self, attribute, value):
        # If a quantity is passed, it must match the quantity field
        if isinstance(value, pint.Quantity):
            if self.quantity is None:
                raise ValueError(
                    f"while validating {attribute.name}, got a Pint quantity, "
                    "incompatible with 'quantity' field set to None; "
                    "please provide a unitless value"
                )
            else:
                expected_units = ucc.get(self.quantity)

                if not pinttr.util.units_compatible(expected_units, value.units):
                    raise pinttr.exceptions.UnitsError(
                        value.units,
                        expected_units,
                        extra_msg=f"while validating {attribute.name}, got units "
                        f"'{value.units}' incompatible with quantity {self.quantity} "
                        f"(expected '{expected_units}')",
                    )

    def __init__(
        self,
        id: str | None = None,
        quantity: PhysicalQuantity | str | None = None,
        *,
        value: int | float | pint.Quantity,
    ):
        # If a quantity is set and a unitless value is passed, it is
        # automatically applied appropriate units
        if quantity is not None and not isinstance(value, pint.Quantity):
            value = pinttr.util.ensure_units(value, ucc.get(quantity))

        self.__attrs_init__(id=id, quantity=quantity, value=value)

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return np.full_like(w, self.value.m) * self.value.units

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_mono(w=w)

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        # Convert bounds to unitless values
        wavelength_units = wmin.units
        wmin = wmin.magnitude
        wmax = wmax.m_as(wavelength_units)

        # Compute integral
        integral = self.value * (wmax - wmin)

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
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring

        return {
            "value": SceneParameter(
                func=lambda ctx: float(self.eval(ctx.si).m_as(uck.get(self.quantity))),
                flags=KernelSceneParameterFlags.SPECTRAL,
            )
        }
