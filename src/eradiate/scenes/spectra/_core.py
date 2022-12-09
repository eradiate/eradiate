from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from functools import singledispatchmethod

import attrs
import pint

from ..._factory import Factory
from ...attrs import documented, parse_docs
from ...scenes.core import SceneElement
from ...spectral_index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
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
        (
            "_multi_delta.MultiDeltaSpectrum",
            "multi_delta",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.spectra",
)


@parse_docs
@attrs.define
class Spectrum(SceneElement, ABC):
    """
    Spectrum abstract base class.
    """

    quantity: PhysicalQuantity = documented(
        attrs.field(
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

    @singledispatchmethod
    def eval(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate spectrum at a given spectral index.
        
        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        value : quantity
            Evaluated spectrum.
        
        Notes
        -----
        This method dispatches evaluation to specialised methods depending
        on the spectral index type.
        """
        raise NotImplementedError

    @eval.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_mono(spectral_index.w)

    @eval.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_ckd(spectral_index.w, spectral_index.g)

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
        pass

    @abstractmethod
    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        """
        Evaluate spectrum in CKD modes.

        Parameters
        ----------
        w : quantity
            Spectral bin center wavelength.
        
        g: float
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
