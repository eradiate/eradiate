from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import attr
import pint

import eradiate

from ..._factory import Factory
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...ckd import Bin
from ...contexts import CKDSpectralContext, MonoSpectralContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...scenes.core import SceneElement
from ...units import PhysicalQuantity


class SpectrumFactory(Factory):
    def converter(self, quantity: Any) -> Callable[[Any], Any]:
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
                        return self.convert({**value, "quantity": quantity})
                except KeyError:
                    pass

            return self.convert(value)

        return f


spectrum_factory = SpectrumFactory()


@parse_docs
@attr.s
class Spectrum(SceneElement, ABC):
    """
    Spectrum abstract base class.
    """

    quantity: PhysicalQuantity = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(PhysicalQuantity),
            repr=lambda x: str(x),
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

    def eval(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate spectrum based on a spectral context. This method dispatches
        evaluation to specialised methods depending on the active mode.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns → :class:`pint.Quantity`:
            Evaluated spectrum as a scalar.
        """
        assert spectral_ctx is not None  # Testing safeguard for leftover None values

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_ckd(spectral_ctx.bin).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate spectrum in monochromatic modes.

        Parameter ``w`` (:class:`pint.Quantity`):
            Wavelength values at which the spectrum is to be evaluated.

        Returns → :class:`pint.Quantity`:
            Evaluated spectrum as an array with the same shape as ``w``.
        """
        pass

    @abstractmethod
    def eval_ckd(self, *bins: Bin) -> pint.Quantity:
        """
        Evaluate spectrum in CKD modes.

        Parameter ``*bins`` (:class:`.Bin`):
            One or several CKD bins for which to evaluate the spectrum.

        Returns → :class:`pint.Quantity`:
            Evaluated spectrum as an array with shape (len(bins),).
        """
        pass
