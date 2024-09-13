from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from functools import singledispatchmethod

import attrs
import pint

from ..core import NodeSceneElement
from ..._factory import Factory
from ...attrs import define, documented
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import PhysicalQuantity


class SpectrumFactory(Factory):
    def converter(self, quantity: str | PhysicalQuantity) -> t.Callable[[t.Any], t.Any]:
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
          * ``"multi_delta"``;

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
                # Convert unitless values with dict wrapping and recursive call
                return f({"type": "uniform", "quantity": quantity, "value": value})

            if isinstance(value, dict):
                # If generic spectrum is requested without a specified
                # "quantity" field, add pre-configured quantity and attempt
                # conversion using regular conversion protocol
                try:
                    if (
                        value["type"] in {"uniform", "interpolated", "multi_delta"}
                        and "quantity" not in value
                    ):
                        return self.convert({**value, "quantity": quantity})
                except KeyError:
                    # Note: A missing "type" field will also run this case, and
                    # the ill-formed dict will be correctly reported upon
                    # regular conversion
                    pass

            if isinstance(value, Spectrum) and value.quantity is None:
                # If quantity is unspecified, apply the expected quantity
                return self.convert(attrs.evolve(value, quantity=quantity))

            # Regular conversion (happens if value is neither int, float nor
            # dict without "quantity" field)
            return self.convert(value)

        return f


spectrum_factory = SpectrumFactory()
spectrum_factory.register_lazy_batch(
    [
        (
            "_air_scattering_coefficient.AirScatteringCoefficientSpectrum",
            "air_scattering_coefficient",
            {},
        ),
        (
            "_interpolated.InterpolatedSpectrum",
            "interpolated",
            {},
        ),
        (
            "_solar_irradiance.SolarIrradianceSpectrum",
            "solar_irradiance",
            {},
        ),
        (
            "_uniform.UniformSpectrum",
            "uniform",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.spectra",
)


@define(eq=False, slots=False)
class Spectrum(NodeSceneElement, ABC):
    """
    Spectrum interface.

    Notes
    -----
    * This class is to be used as a mixin.
    * Subclasses must implement :meth:`eval_mono`, :meth:`eval_ckd` and
      :meth:`integral`.

    Warnings
    --------
    Leaving the ``quantity`` field unset is discouraged and permitted only to
    allow for more convenience in some parts of the code (*e.g.* to increase the
    flexibility of spectrum conversion protocols).
    """

    quantity: PhysicalQuantity | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(PhysicalQuantity),
            repr=lambda x: "None" if x is None else x.value.upper(),
        ),
        doc="Physical quantity which the spectrum represents. The specified "
        "quantity must be one which varies with wavelength. "
        "See :meth:`.PhysicalQuantity.spectrum` for allowed values.\n"
        "\n"
        "Child classes should implement value units validation and conversion "
        "based on ``quantity``.",
        type=".PhysicalQuantity or None",
        init_type=".PhysicalQuantity or str, optional",
        default="None",
    )

    @quantity.validator
    def _quantity_validator(self, attribute, value):
        if value is None:
            return

        if value not in PhysicalQuantity.spectrum():
            raise ValueError(
                f"while validating {attribute.name}: "
                f"got value '{value}', expected one of "
                f"{str(PhysicalQuantity.spectrum())}"
            )

    @singledispatchmethod
    def eval(self, si: SpectralIndex) -> pint.Quantity:
        """
        Evaluate spectrum at a given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        value : quantity
            Evaluated spectrum.

        Notes
        -----
        This method dispatches evaluation to specialized methods depending
        on the spectral index type.
        """
        raise NotImplementedError

    @eval.register(MonoSpectralIndex)
    def _(self, si) -> pint.Quantity:
        return self.eval_mono(w=si.w)

    @eval.register(CKDSpectralIndex)
    def _(self, si) -> pint.Quantity:
        return self.eval_ckd(w=si.w, g=si.g)

    @abstractmethod
    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate spectrum in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength.

        Returns
        -------
        value : quantity
            Evaluated spectrum as an array with the same shape as ``w``.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        """
        Evaluate spectrum in CKD modes.

        Parameters
        ----------
        w : quantity
            Spectral bin center wavelength.

        g : float
            Absorption coefficient cumulative probability.

        Returns
        -------
        value : quantity
            Evaluated spectrum as an array with shape ``w``.

        Notes
        -----
        It is assumed that ``w`` and ``g`` have the same shape.
        In CKD mode, it is assumed that all spectra—except that of the
        absorption coefficient—are uniform over the spectral bin. These
        spectra are evaluated at the spectral bin center wavelength.
        """
        pass

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
        value : quantity or ndarray
            Computed integral value, in units consistent with the specified
            quantity.

        Warnings
        --------
        If the ``quantity`` field is unset (*i.e.* left to its default value
        ``None``), the output of this method will inherit the units of value
        fields (the actual policy depends on the implementation, and unitless
        values are intepreted as dimnesionless). Note that leaving the
        ``quantity`` field unset is discouraged.
        """
        raise NotImplementedError
