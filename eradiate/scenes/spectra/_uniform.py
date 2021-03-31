import attr
import pint
import pinttr
from pint import DimensionalityError

from ... import unit_context_config as ucc
from ... import unit_context_kernel as uck
from ..._attrs import documented, parse_docs
from ..._units import PhysicalQuantity
from ...scenes.spectra import Spectrum, SpectrumFactory
from ...validators import is_positive


@SpectrumFactory.register("uniform")
@parse_docs
@attr.s
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum (*i.e.* constant against wavelength). Supports basic arithmetics.
    """

    value = documented(
        attr.ib(default=1.0),
        doc="Uniform spectrum value. If a float is passed and ``quantity`` is not "
        "``None``, it is automatically converted to appropriate configuration "
        "default units. If a :class:`~pint.Quantity` is passed and ``quantity`` "
        "is not ``None``, units will be checked for consistency.",
        type="float or :class:`~pint.Quantity`",
    )

    @value.validator
    def value_validator(self, attribute, value):
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

        is_positive(self, attribute, value)

    def __attrs_post_init__(self):
        if self.quantity is not None and self.value is not None:
            self.value = pinttr.converters.ensure_units(
                self.value, ucc.get(self.quantity)
            )

    @property
    def _values(self):
        return self.value

    @property
    def values(self):
        return self.value

    def __add__(self, other):
        # Preserve quantity field only if it is the same for both operands
        if self.quantity is other.quantity:
            quantity = self.quantity
        else:
            quantity = None

        try:
            value = self.value + other.value
        except DimensionalityError as e:
            raise pinttr.exceptions.UnitsError(e.units1, e.units2)

        return UniformSpectrum(quantity=quantity, value=value)

    def __sub__(self, other):
        # Preserve quantity field only if it is the same for both
        # operands
        if self.quantity is other.quantity:
            quantity = self.quantity
        else:
            quantity = None

        try:
            value = self.value - other.value
        except DimensionalityError as e:
            raise pinttr.exceptions.UnitsError(e.units1, e.units2)

        return UniformSpectrum(quantity=quantity, value=value)

    def __mul__(self, other):
        # We can only preserve 'dimensionless', other quantities are much
        # more challenging to infer
        if (
            self.quantity is PhysicalQuantity.DIMENSIONLESS
            and other.quantity is PhysicalQuantity.DIMENSIONLESS
        ):
            quantity = PhysicalQuantity.DIMENSIONLESS
        else:
            quantity = None

        try:
            value = self.value * other.value
        except DimensionalityError as e:
            raise pinttr.exceptions.UnitsError(e.units1, e.units2)

        return UniformSpectrum(quantity=quantity, value=value)

    def __truediv__(self, other):
        # We can only infer 'dimensionless' if both operands have the same
        # quantity field, other cases are much more challenging
        if self.quantity is other.quantity and self.quantity is not None:
            quantity = PhysicalQuantity.DIMENSIONLESS
        else:
            quantity = None

        try:
            value = self.value / other.value
        except DimensionalityError as e:
            raise pinttr.exceptions.UnitsError(e.units1, e.units2)

        return UniformSpectrum(quantity=quantity, value=value)

    def kernel_dict(self, ref=True):
        kernel_units = uck.get(self.quantity)

        return {
            "spectrum": {
                "type": "uniform",
                "value": self.value.m_as(kernel_units),
            }
        }
