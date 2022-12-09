from __future__ import annotations

import logging
import typing as t

import attrs
import numpy as np
import numpy.typing as npt
import pint
import pinttr
import xarray as xr

from .attrs import documented, parse_docs
from .scenes.spectra import InterpolatedSpectrum, MultiDeltaSpectrum
from .scenes.spectra._interpolated import where_non_zero
from .spectral_index import MonoSpectralIndex
from .units import to_quantity
from .units import unit_context_config as ucc
from .units import unit_registry as ureg

logger = logging.getLogger(__name__)

class Select:
    def __init__(self, selected: t.Callable[[pint.Quantity], npt.ArrayLike[bool]]):
        self.selected = selected

    def __call__(self, w: pint.Quantity) -> bool:
        return self.selected(w)
    
    @classmethod
    def included(
        cls, 
        wmin: pint.Quantity,
        wmax: pint.Quantity,
    ) -> Select:
        return cls(selected=included(wmin=wmin, wmax=wmax))
    

def included(
    wmin: pint.Quantity,
    wmax: pint.Quantity,
) -> t.Callable[[pint.Quantity], npt.ArrayLike[bool]]:

    wmins = np.atleast_1d(wmin)
    wmaxs = np.atleast_1d(wmax)
    
    def selected(w: pint.Quantity):
        mask = np.zeros_like(w, dtype=bool)
        for _wmin, _wmax in zip(wmins, wmaxs):
            mask |= (_wmin <= w) & (w <= _wmax)

        return mask
    
    return selected



@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class WavelengthSet:
    """
    A data class representing a wavelength set used in monochromatic mode.

    See Also
    --------
    :class:`.ckd.BinSet`

    Notes
    -----
    This is class is a simple container for an array of wavelengths at which
    a monochromatic experiment is to be performed.
    Its design is inspired by :class:`.ckd.BinSet`.
    The `select_from_srf` method is used to select a subset of (or create new)
    wavelengths based on a spectral response function.
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

    def select_with(self, spectrum) -> WavelengthSet:
        """
        Notes
        -----
        If the type of the passed ``spectrum`` is :class:`.MultipleDeltaSpectrum`,
        the returned wavelength set contains exactly the wavelengths defined
        by the ``spectrum``. In other words, this is not a selection but a 
        conversion.
        """
        if isinstance(spectrum, MultiDeltaSpectrum):
            return WavelengthSet(spectrum.w)

        elif isinstance(spectrum, InterpolatedSpectrum):
            wranges = where_non_zero(spectrum)
            wranges_sizes = np.array([wrange.size for wrange in wranges])
            if np.any(wranges_sizes == 1):
                print(wranges)
                msg = (
                    "The support of this spectrum contains ranges of length"
                    " 1, which is not supported by the monochromatic mode."
                )
                logger.critical(msg)
                raise ValueError(msg)
            else:
                wmin = np.stack([wrange[0] for wrange in wranges])
                wmax = np.stack([wrange[-1] for wrange in wranges])
                print(f"wmin: {wmin} wmax: {wmax}")
                select = Select.included(wmin, wmax)
                selected = select(self.wavelengths)
                selected_wavlenghts = self.wavelengths[selected]

            return WavelengthSet(selected_wavlenghts)

        else:
            raise NotImplementedError(
                f"Spectrum must be of type MultipleDeltaSpectrum or "
                f"InterpolatedSpectrum (got {type(spectrum)})"
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
        dataset : :class:`xarray.Dataset`
            Absorption dataset.

        Returns
        -------
        :class:`.WavelengthSet`
            Generated wavelength set.
        """
        return cls(wavelengths=to_quantity(dataset.w))

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
        start : :class:`pint.Quantity`
            First wavelength.

        stop : :class:`pint.Quantity`
            Last wavelength.

        step : :class:`pint.Quantity`
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
            ) * wunits
        )

