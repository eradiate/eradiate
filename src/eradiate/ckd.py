from __future__ import annotations

import logging
import typing as t
from itertools import compress

import attrs
import numpy as np
import numpy.typing as npt
import pint
import pinttr
import xarray as xr

from .attrs import documented, parse_docs
from .quad import Quad
from .scenes.spectra import InterpolatedSpectrum, MultiDeltaSpectrum, Spectrum
from .scenes.spectra._interpolated import where_non_zero
from .spectral_index import CKDSpectralIndex
from .units import unit_context_config as ucc
from .units import unit_registry as ureg

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
#                              CKD bin data classes
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class Bin:
    """
    A data class representing a spectral bin in CKD modes.
    """

    wmin: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc='Bin lower spectral bound.\n\nUnit-enabled field (default: ucc["wavelength"]).',
        type="quantity",
        init_type="quantity or float",
    )

    wmax: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        doc='Bin upper spectral bound.\n\nUnit-enabled field (default: ucc["wavelength"]).',
        type="quantity",
        init_type="quantity or float",
    )

    @wmin.validator
    @wmax.validator
    def _wbounds_validator(self, attribute, value):
        if not self.wmin < self.wmax:
            raise ValueError(
                f"while validating {attribute.name}: wmin must be lower than wmax"
            )

    quad: Quad = documented(
        attrs.field(
            default=Quad.gauss_legendre(2),
            repr=lambda x: x.str_summary, validator=attrs.validators.instance_of(Quad)
        ),
        doc="Quadrature rule attached to the CKD bin.",
        type=":class:`.Quad`",
    )

    @property
    def width(self) -> pint.Quantity:
        """quantity : Bin spectral width."""
        return self.wmax - self.wmin

    @property
    def wcenter(self) -> pint.Quantity:
        """quantity : Bin central wavelength."""
        return 0.5 * (self.wmin + self.wmax)
    
    def pretty_repr(self) -> str:
        """str : Pretty representation of the bin."""
        units = ureg.Unit("nm")
        wrange = f"[{self.wmin.m_as(units):.1f}, {self.wmax.m_as(units):.1f}] {units:~P}"
        quad = self.quad.pretty_repr()
        return f"{wrange}: {quad}"

    def spectral_indices(self) -> t.Generator[CKDSpectralIndex]:
        for _, value in enumerate(self.quad.eval_nodes(interval=[0.0, 1.0])):
            yield CKDSpectralIndex(w=self.wcenter, g=value)


# ------------------------------------------------------------------------------
#                               Bin selection
# ------------------------------------------------------------------------------

class Select:

    def __init__(self, selected: t.Callable[[t.List[Bin]], npt.ArrayLike[bool]]):
        self.selected = selected

    def __call__(self, b: t.List[Bin]) -> npt.ArrayLike[bool]:
        return self.selected(b)
    
    @classmethod
    def includes(
        cls, w: pint.Quantity
    ) -> Select:
        return cls(selected=includes(w))
    
    @classmethod
    def overlaps(
        cls, wmin: pint.Quantity, wmax: pint.Quantity
    ) -> Select:
        return cls(selected=overlaps(wmin, wmax))

def includes(w: pint.Quantity) -> t.Callable[[t.List[Bin]], npt.ArrayLike[bool]]:

    def selected(b: t.List[Bin]) -> npt.ArrayLike[bool]:
        b = [b] if isinstance(b, Bin) else b
        mask = np.zeros(len(b), dtype=bool)
        for _w in np.atleast_1d(w):
            binwmin = np.stack([bin.wmin for bin in b])
            binwmax = np.stack([bin.wmax for bin in b])
            mask |= (binwmin <= _w) & (_w < binwmax)
        return mask

    return selected

def overlaps(
    wmin: pint.Quantity, wmax: pint.Quantity
) -> t.Callable[[t.List[Bin]], npt.ArrayLike[bool]]:

    def selected(b: t.List[Bin]) -> npt.ArrayLike[bool]:
        b = [b] if isinstance(b, Bin) else b
        mask = np.zeros(len(b), dtype=bool)
        for _wmin, _wmax in zip(np.atleast_1d(wmin), np.atleast_1d(wmax)):
            binwmin = np.stack([bin.wmin for bin in b])
            binwmax = np.stack([bin.wmax for bin in b])
            mask |= (binwmin < _wmax) & (_wmin < binwmax)
        return mask

    return selected


# ------------------------------------------------------------------------------
#                              Bin set data class
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define(eq=False, frozen=True, slots=True)
class BinSet:
    """
    A data class representing a bin set used in CKD mode.
    """
    bins: t.List[Bin] = documented(
        attrs.field(
            converter=list,
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(Bin)
            ),
            repr=lambda x: "\n        " + "\n        ".join([
                    bin.pretty_repr()
                    for bin in sorted(x, key=lambda b: b.wcenter.m_as("nm"))
                ]) + "\n",
        ),
        doc="Set of bins.",
        type="set of :class:`.Bin`",
        init_type="iterable of :class:`.Bin`",
    )

    def select_with(self, spectrum: Spectrum) -> BinSet:
        """
        Returns a copy of the bin set containing only the bins that are
        selected by the given spectrum.

        Parameters
        ----------
        spectrum : :class:`.Spectrum`
            Spectrum.

        Returns
        -------
        :class:`.BinSet`
            Selected bins.
        """
        bins = self.bins
        if isinstance(spectrum, InterpolatedSpectrum):
            selects = self._get_selects(spectrum)
            selected = np.zeros(len(bins), dtype=bool)
            for select in selects:
                selected |= select(bins)
        elif isinstance(spectrum, MultiDeltaSpectrum):
            select = Select.includes(spectrum.wavelengths)
            selected = select(bins)
        else:
            raise NotImplementedError(
                f"Spectrum type must be InterpolatedSpectrum or "
                f"MultipleDeltaSpectrum (got {type(spectrum)})"
            )
        return BinSet(bins=list(compress(bins, selected)))
    

    def _get_selects(self, spectrum: InterpolatedSpectrum):
        """
        Helper function for :meth:`.select_with` for InterpolatedSpectrum.
        """
        selects = []
        wranges = where_non_zero(spectrum)

        # Isolated wavelength points
        ws = [wrange[0] for wrange in wranges if wrange.size == 1]
        if len(ws) > 0:
            ws = np.stack(ws)
            selects.append(Select.includes(ws))

        # Wavelength intervals
        wmins = [wrange[0] for wrange in wranges if wrange.size > 1]
        wmaxs = [wrange[-1] for wrange in wranges if wrange.size > 1]
        if len(wmins) > 0:
            wmins = np.stack(wmins)
            wmaxs = np.stack(wmaxs)
            selects.append(Select.overlaps(wmins, wmaxs))
        
        return selects

    def spectral_indices(self) -> t.Generator[CKDSpectralIndex]:
        for bin in self.bins:
            yield from bin.spectral_indices()

    @classmethod
    def arange(
        cls,
        start: pint.Quantity,
        stop: pint.Quantity,
        step: pint.Quantity = 10.0 * ureg.nm,
        quad: Quad = Quad.gauss_legendre(2),
    ) -> BinSet:
        """
        Generate a bin set with linearly spaced bins.

        Parameters
        ----------
        start : quantity
            Lower bound of first bin.

        stop : quantity
            Upper bound of last bin.

        step : quantity
            Bin width.
        
        quad : :class:`.Quad`
            Quadrature rule (same for all bins in the set).

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        bins = []

        wunits = ucc.get("wavelength")
        _start = start.m_as(wunits)
        _stop = stop.m_as(wunits)
        _step = step.m_as(wunits)

        for wmin in np.arange(_start, _stop, _step):
            wmax = wmin + _step
            bins.append(Bin(
                wmin=wmin * wunits,
                wmax=wmax * wunits,
                quad=quad,
            ))

        return cls(bins)

    @classmethod
    def from_wavelengths(
        cls,
        wavelengths: pint.Quantity,
        width: pint.Quantity = 10.0 * ureg.nm,
        quad: Quad = Quad.gauss_legendre(2),
    ) -> BinSet:
        """
        Generate a bin set with bins centered on the given wavelengths.

        Parameters
        ----------
        wavelengths : sequence of quantity
            Wavelengths to center bins on.

        quad : :class:`.Quad`
            Quadrature rule (same for all bins in the set).

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        """
        bins = []

        for wcenter in np.atleast_1d(wavelengths):
            bins.append(Bin(
                wmin=wcenter - width / 2,
                wmax=wcenter + width / 2,
                quad=quad,
            ))

        return cls(bins)

    @classmethod
    def from_absorption_dataset(
        cls,
        dataset: xr.Dataset,
        quad: Quad = Quad.gauss_legendre(8),
    ) -> BinSet:
        """
        Generate a bin set from an absorption dataset.

        Parameters
        ----------
        dataset : :class:`xarray.Dataset`
            Absorption dataset.

        quad : :class:`.Quad`
            Quadrature rule (same for all bins in the set).

        Returns
        -------
        :class:`.BinSet`
            Generated bin set.
        
        Notes
        -----
        Assumes that:
        * the absorption dataset has a ``bin`` coordinate with values in
          nanometers.
        * the absorption dataset has a ``bin_set`` attribute with the bin
          width in nanometers.
        """
        wavelengths = np.unique(np.array(dataset.bin.values, dtype=float)) * ureg.nm
        width = ureg(dataset.attrs["bin_set"])
        return cls.from_wavelengths(wavelengths, width=width, quad=quad)


def resample(
    spectrum: InterpolatedSpectrum,
    binset: BinSet,
) -> InterpolatedSpectrum:
    """
    Resample an interpolated spectrum on a binset.

    Parameters
    ----------
    spectrum : :class:`.InterpolatedSpectrum`
        Interpolated spectrum to resample.

    binset : :class:`.BinSet`
        Bin set to resample to.

    Returns
    -------
    :class:`.InterpolatedSpectrum`
        New spectrum whose wavelength points are aligned with the bin centers
        and values are the average values of the original spectrum over each
        bin.
    """
    binset = binset.select_with(spectrum)
    wavelengths = np.full_like(binset.bins, np.nan)
    values = np.full_like(binset.bins, np.nan)
    for i, bin in enumerate(binset.bins):
        wavelengths[i] = bin.wcenter
        values[i] = spectrum.integral(bin.wmin, bin.wmax) / bin.width
    return InterpolatedSpectrum(
        id=spectrum.id,
        quantity=spectrum.quantity,
        wavelengths=wavelengths,
        values=values,
    )
