"""Spectrum-related scene generation facilities.

.. admonition:: Registered factory members [:class:`SpectrumFactory`]
   :class: hint

   .. factorytable::
      :factory: SpectrumFactory
"""
from abc import ABC

import attr
import numpy as np
import pint
import pinttr
from pint import DimensionalityError

import eradiate
from .core import SceneElement
from .. import data
from .._attrs import (
    documented,
    parse_docs,
)
from .._factory import BaseFactory
from .._units import PhysicalQuantity
from .._units import unit_context_config as ucc
from .._units import unit_context_kernel as uck
from .._units import unit_registry as ureg
from ..exceptions import ModeError
from ..validators import is_positive


@parse_docs
@attr.s
class Spectrum(SceneElement, ABC):
    """Spectrum abstract base class."""

    quantity = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(PhysicalQuantity),
        ),
        doc="Physical quantity which the spectrum represents. If not ``None``, "
            "the specified quantity must be one which varies with wavelength. "
            "See :meth:`.PhysicalQuantity.spectrum` for allowed values.\n"
            "\n"
            "Child classes should implement value units validation and conversion "
            "based on ``quantity``. In particular, no unit validation or conversion "
            "should occur if ``quantity`` is ``None``.",
        type="str or :class:`.PhysicalQuantity` or None",
    )

    @quantity.validator
    def _quantity_validator(self, attribute, value):
        if value is None:
            return

        if value not in PhysicalQuantity.spectrum():
            raise ValueError(
                f"while validating {attribute.name}: "
                f"got value '{value}', expected one of {str()}"
            )

    @property
    def _values(self):
        """Return spectrum internal values."""
        raise NotImplementedError

    @property
    def values(self):
        """Evaluate (as a Pint quantity) spectrum values based on currently active mode."""
        raise NotImplementedError


class SpectrumFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`Spectrum`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: SpectrumFactory
    """

    _constructed_type = Spectrum
    registry = {}

    @staticmethod
    def converter(quantity):
        """Generate a converter wrapping :meth:`SpectrumFactory.convert` to
        handle defaults for shortened spectrum definitions. The produced
        converter processes a parameter ``value`` as follows:

        * if ``value`` is a float or a :class:`pint.Quantity`, the converter
          calls itself using a dictionary
          ``{"type": "uniform", "quantity": quantity, "value": value}``;
        * if ``value`` is a dictionary, it adds a ``"quantity": quantity`` entry
          for the following values of the ``"type"`` entry:
          * ``"uniform"``;
        * otherwise, it forwards ``value`` to
          :meth:`.SpectrumFactory.convert`.

        Parameter ``quantity`` (str or :class:`PhysicalQuantity`):
            Quantity specifier (converted by :meth:`SpectrumQuantity.from_any`).
            See :meth:`PhysicalQuantity.spectrum` for suitable values.

        Returns → callable:
            Generated converter.
        """

        def f(value):
            if isinstance(value, (float, pint.Quantity)):
                return f({
                    "type": "uniform",
                    "quantity": quantity,
                    "value": value
                })

            if isinstance(value, dict):
                try:
                    if value["type"] == "uniform" and "quantity" not in value:
                        return SpectrumFactory.convert(
                            {**value, "quantity": quantity}
                        )
                except KeyError:
                    pass

            return SpectrumFactory.convert(value)

        return f


@SpectrumFactory.register("uniform")
@parse_docs
@attr.s
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum (*i.e.* constant against wavelength). Supports basic
    arithmetics.
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
                    f"(expected '{expected_units}')"
                )

        is_positive(self, attribute, value)

    def __attrs_post_init__(self):
        if self.quantity is not None and self.value is not None:
            self.value = pinttr.converters.ensure_units(self.value, ucc.get(self.quantity))

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
        if self.quantity is PhysicalQuantity.DIMENSIONLESS \
                and other.quantity is PhysicalQuantity.DIMENSIONLESS:
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


@SpectrumFactory.register("solar_irradiance")
@parse_docs
@attr.s(frozen=True)
class SolarIrradianceSpectrum(Spectrum):
    """Solar irradiance spectrum scene element [:factorykey:`solar_irradiance`].

    This scene element produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this element is controlled by the ``dataset`` attribute (see
    :mod:`eradiate.data.solar_irradiance_spectra` for available data sets).

    The spectral range of the data sets shipped can vary and an attempt for use
    outside of the supported spectral range will raise a :class:`ValueError`
    upon calling :meth:`kernel_dict`.

    The generated kernel dictionary varies based on the selected mode of
    operation. The ``scale`` parameter can be used to adjust the value based on
    unit conversion or to account for variations of the Sun-planet distance.

    The produced kernel dictionary automatically adjusts its irradiance units
    depending on the selected kernel default units.
    """

    #: Physical quantity
    quantity = attr.ib(
        default=PhysicalQuantity.IRRADIANCE,
        init=False,
        repr=False
    )

    dataset = documented(
        attr.ib(
            default="thuillier_2003",
            validator=attr.validators.instance_of(str),
        ),
        doc="Dataset identifier. Allowed values: see "
        ":attr:`solar irradiance dataset documentation <eradiate.data.solar_irradiance_spectra>`. "
        "Default: ``\"thuillier_2003\"``. ",
        type="str",
    )

    scale = documented(
        attr.ib(
            default=1.0,
            converter=float,
            validator=is_positive,
        ),
        doc="Scaling factor. Default: 1.",
        type="float",
    )

    @dataset.validator
    def _dataset_validator(self, attribute, value):
        if value not in data.registered("solar_irradiance_spectrum"):
            raise ValueError(
                f"while setting {attribute.name}: '{value}' not in "
                f"list of supported solar irradiance spectra "
                f"{data.registered('solar_irradiance_spectrum')}"
            )

    data = attr.ib(
        init=False,
        repr=False
    )

    @data.default
    def _data_factory(self):
        # Load dataset
        try:
            return data.open("solar_irradiance_spectrum", self.dataset)
        except KeyError:
            raise ValueError(f"unknown dataset {self.dataset}")

    @property
    def values(self):
        if eradiate.mode().is_monochromatic():
            w = eradiate.mode().wavelength
            w_units = self.data.w.attrs["units"]
            ssi_units = self.data.ssi.attrs["units"]
            return ureg.Quantity(
                self.data.ssi.interp(w=w.m_as(w_units)).data.squeeze(), ssi_units
            )
        else:
            raise ModeError(f"unsupported mode '{eradiate.mode()}'")

    def kernel_dict(self, ref=True):
        mode = eradiate.mode()

        if mode.is_monochromatic():
            wavelength = mode.wavelength.to(ureg.nm).magnitude

            if self.dataset == "solid_2017":
                raise NotImplementedError(
                    "Solar irradiance spectrum datasets with a non-empty time "
                    "coordinate are not supported yet."
                )
            # TODO: add support to solar irradiance spectrum datasets with a
            #  non-empty time coordinate

            irradiance_magnitude = float(
                self.data.ssi.interp(
                    w=wavelength,
                    method="linear",
                ).values
            )

            # Raise if out of bounds or ill-formed dataset
            if np.isnan(irradiance_magnitude):
                raise ValueError(f"dataset evaluation returned nan")

            # Apply units
            irradiance = ureg.Quantity(
                irradiance_magnitude,
                self.data.ssi.attrs["units"]
            )

            # Apply scaling, build kernel dict
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": irradiance.to(uck.get("irradiance")).magnitude *
                             self.scale
                }
            }

        else:
            raise ModeError(f"unsupported mode '{mode.id}'")
