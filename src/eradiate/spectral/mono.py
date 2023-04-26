from __future__ import annotations

import logging
import typing as t

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

from .index import MonoSpectralIndex
from ..attrs import documented, parse_docs
from ..constants import SPECTRAL_RANGE_MAX, SPECTRAL_RANGE_MIN
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class WavelengthSet:
    """
    A data class representing a wavelength set used in monochromatic modes.

    See Also
    --------
    :class:`~.BinSet`

    Notes
    -----
    This is class is a simple container for an array of wavelengths at which
    a monochromatic experiment is to be performed.
    Its design is inspired by :class:`~.BinSet`.
    """

    wavelengths: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            on_setattr=None,
            converter=np.atleast_1d,
        ),
        doc="Wavelengths.",
        type="quantity",
        init_type="quantity or array-like or float",
    )

    def spectral_indices(self) -> t.Generator[MonoSpectralIndex]:
        for w in self.wavelengths:
            yield MonoSpectralIndex(w=w)

    @classmethod
    def from_absorption_dataset(cls, dataset: xr.Dataset) -> WavelengthSet:
        """
        Create a wavelength set from an absorption dataset.

        Parameters
        ----------
        dataset : Dataset
            Absorption dataset.

        Returns
        -------
        :class:`.WavelengthSet`
            Generated wavelength set.
        """
        # we dont know if the absorption dataset 'w' variable holds
        # wavelengths or wavenumbers, so we check the units
        w = to_quantity(dataset.w)
        if w.check("[length]^-1"):
            wavelengths = np.sort(1 / w).to(
                "nm"
            )  # ordered wavenumbers are reversed in wavelength-space
        elif w.check("[length]"):
            wavelengths = w

        return cls(wavelengths=wavelengths)

    @classmethod
    def arange(
        cls,
        start: pint.Quantity,
        stop: pint.Quantity,
        step: pint.Quantity,
    ) -> WavelengthSet:
        """
        Create a wavelength set from an array of wavelengths.

        Parameters
        ----------
        start : quantity
            First wavelength.

        stop : quantity
            Last wavelength.

        step : quantity
            Wavelength step.

        Returns
        -------
        :class:`.WavelengthSet`
            Generated wavelength set.
        """
        wunits = ucc.get("wavelength")
        return cls(
            wavelengths=np.arange(
                start.m_as(wunits),
                stop.m_as(wunits),
                step.m_as(wunits),
            )
            * wunits
        )

    @classmethod
    def default(cls) -> WavelengthSet:
        """
        Generate a default wavelength set, which covers Eradiate's default
        spectral range with 1 nm spacing.
        """
        dw = 1.0 * ureg.nm
        return cls.arange(
            start=SPECTRAL_RANGE_MIN,
            stop=SPECTRAL_RANGE_MAX + dw,
            step=dw,
        )
