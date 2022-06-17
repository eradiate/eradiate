from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attr
import pint

import eradiate

from ..._factory import Factory
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import SpectralContext
from ...exceptions import UnsupportedModeError
from ...scenes.core import SceneElement
from ...units import PhysicalQuantity


class SpectrumFactory(Factory):
    def converter(
        self, quantity: t.Union[str, PhysicalQuantity]
    ) -> t.Callable[[t.Any], t.Any]:
        """
        Generate a converter wrapping :meth:`SpectrumFactory.convert` to
        handle defaults for shortened spectrum definitions. The produced
        converter processes a parameter ``value`` as follows:

        * if ``value`` is an int, a float or a :class:`pint.Quantity`, the
          converter calls itself using a dictionary
          ``{"type": "uniform", "quantity": quantity, "value": value}``;
        * if ``value`` is a dictionary, it adds a ``"quantity": quantity`` entry
          for the following values of the ``"type"`` entry:

          * ``"uniform"``;
          * ``"interpolated"``;

        * otherwise, it forwards ``value`` to :meth:`.SpectrumFactory.convert`.

        Parameters
        ----------
        quantity : str or .PhysicalQuantity
            Quantity specifier (converted by :class:`.PhysicalQuantity`).
            See :meth:`.PhysicalQuantity.spectrum` for suitable values.

        Returns
        -------
        callable
            Generated converter.
        """

        def f(value):
            if isinstance(value, (int, float, pint.Quantity)):
                # Convert quantity-less values with dict wrapping and recursive call
                return f({"type": "uniform", "quantity": quantity, "value": value})

            if isinstance(value, dict):
                # If generic spectrum is requested without a specified
                # "quantity" field, add pre-configured quantity and attempt
                # conversion using regular conversion protocol
                try:
                    if (
                        value["type"] in {"uniform", "interpolated"}
                        and "quantity" not in value
                    ):
                        return self.convert({**value, "quantity": quantity})
                except KeyError:
                    # Note: A missing "type" field will also run this case, and
                    # the ill-formed dict will be correctly reported upon
                    # regular conversion
                    pass

            # Regular conversion (happens if value is neither int, float nor
            # dict without "quantity" field)
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
            default="dimensionless",
            converter=PhysicalQuantity,
            repr=lambda x: f"{x.value.upper()}",
        ),
        doc="Physical quantity which the spectrum represents. The specified "
        "quantity must be one which varies with wavelength. "
        "See :meth:`.PhysicalQuantity.spectrum` for allowed values.\n"
        "\n"
        "Child classes should implement value units validation and conversion "
        "based on ``quantity``.",
        type=":class:`.PhysicalQuantity`",
        init_type=":class:`.PhysicalQuantity` or str",
        default="dimensionless",
    )

    @quantity.validator
    def _quantity_validator(self, attribute, value):
        if value not in PhysicalQuantity.spectrum():
            raise ValueError(
                f"while validating {attribute.name}: "
                f"got value '{value}', expected one of {str(PhysicalQuantity.spectrum())}"
            )

    def eval(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate spectrum based on a spectral context. This method dispatches
        evaluation to specialised methods depending on the active mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        value : quantity
            Evaluated spectrum as a scalar.
        """
        if eradiate.mode().is_mono:
            return self.eval_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate spectrum in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength values at which the spectrum is to be evaluated.

        Returns
        -------
        value : quantity
            Evaluated spectrum as an array with the same shape as ``w``.
        """
        pass

    @abstractmethod
    def eval_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        """
        Evaluate spectrum in CKD modes.

        Parameters
        ----------
        *bindexes : .Bindex
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        value : quantity
            Evaluated spectrum as an array with shape ``(len(bindexes),)``.
        """
        pass

    @abstractmethod
    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        """
        Compute the integral of the spectrum on a given interval.

        Parameters
        ----------
        wmin : quantity
            Integration interval's lower bound.

        wmax : quantity
            Integration interval's upper bound.

        Returns
        -------
        value : quantity
            Computed integral value.
        """
        pass
