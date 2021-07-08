from abc import ABC, abstractmethod

import attr
import pint
from dessinemoi import Factory

from ..._factory import BaseFactory
from ...attrs import documented, parse_docs
from ...contexts import SpectralContext
from ...scenes.core import SceneElement
from ...units import PhysicalQuantity


@parse_docs
@attr.s
class Spectrum(SceneElement, ABC):
    """
    Spectrum abstract base class.
    """

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
                f"got value '{value}', expected one of {str(PhysicalQuantity.spectrum())}"
            )

    @abstractmethod
    def eval(self, spectral_ctx: SpectralContext = None) -> pint.Quantity:
        """
        Evaluate spectrum.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Evaluated spectrum.
        """
        pass


class SpectrumFactory(Factory):
    @staticmethod
    def converter(quantity):
        """
        Generate a converter wrapping :meth:`SpectrumFactory.convert` to
        handle defaults for shortened spectrum definitions. The produced
        converter processes a parameter ``value`` as follows:

        * if ``value`` is a float or a :class:`pint.Quantity`, the converter
          calls itself using a dictionary
          ``{"type": "uniform", "quantity": quantity, "value": value}``;
        * if ``value`` is a dictionary, it adds a ``"quantity": quantity`` entry
          for the following values of the ``"type"`` entry:

          * ``"uniform"``;
          * ``"interpolated"``;

        * otherwise, it forwards ``value`` to :meth:`.SpectrumFactory.convert`.

        Parameter ``quantity`` (str or :class:`.PhysicalQuantity`):
            Quantity specifier (converted by :class:`.PhysicalQuantity`).
            See :meth:`.PhysicalQuantity.spectrum` for suitable values.

        Returns → callable:
            Generated converter.
        """

        def f(value):
            if isinstance(value, (float, pint.Quantity)):
                return f({"type": "uniform", "quantity": quantity, "value": value})

            if isinstance(value, dict):
                try:
                    if (
                        value["type"] in {"uniform", "interpolated"}
                        and "quantity" not in value
                    ):
                        return SpectrumFactory.convert({**value, "quantity": quantity})
                except KeyError:
                    pass

            return SpectrumFactory.convert(value)

        return f


spectrum_factory = SpectrumFactory()
