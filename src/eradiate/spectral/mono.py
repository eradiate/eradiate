from __future__ import annotations

import logging
import typing as t

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

from .index import MonoSpectralIndex
from .spectral_set import SpectralSet
from ..attrs import documented, parse_docs
from ..constants import SPECTRAL_RANGE_MAX, SPECTRAL_RANGE_MIN
from ..radprops import MonoAbsorptionDatabase
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class WavelengthSet(SpectralSet):
    """
    A data class representing a wavelength set used in monochromatic modes.
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
    def from_absorption_database(cls, abs_db: MonoAbsorptionDatabase) -> WavelengthSet:
        """
        Create a wavelength set from an absorption dataset.

        Parameters
        ----------
        abs_db : .MonoAbsorptionDatabase
            Absorption dataset.

        Returns
        -------
        .WavelengthSet
        """
        wavelengths = (
            np.sort(abs_db.spectral_coverage.index.get_level_values(1).values) * ureg.nm
        )
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
    def from_srf(
        cls,
        srf: xr.Dataset,
        step: pint.Quantity = 10.0 * ureg.nm,
    ) -> WavelengthSet:
        """
        Generate a wavelength set with linearly spaced bins, that covers the
        spectral range of a spectral response function.

        Parameters
        ----------
        srf: Dataset
            Spectral response function dataset.

        step : quantity
            Wavelength step.
        """
        wavelengths = to_quantity(srf.w)
        wmin = wavelengths.min()
        wmax = wavelengths.max()

        return cls.arange(
            start=wmin - step,
            stop=wmax + step,
            step=step,
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
