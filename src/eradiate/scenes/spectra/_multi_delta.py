from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr

from ._core import Spectrum
from ...attrs import documented, parse_docs
from ...spectral.ckd import BinSet
from ...spectral.mono import WavelengthSet
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def wavelengths_converter(value):
    return np.sort(np.atleast_1d(value))


@parse_docs
@attrs.define(eq=False, slots=False)
class MultiDeltaSpectrum(Spectrum):
    """
    A spectrum made of multiple translated Dirac delta [``multi_delta``].

    Notes
    -----
    This spectrum is for special use.
    It does not have a kernel representation.
    The spectrum cannot be evaluated.
    As a result, neither can it be plotted.
    """

    wavelengths: pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=uck.deferred("wavelength"),
            converter=[
                wavelengths_converter,
                pinttr.converters.to_units(uck.deferred("wavelength")),
            ],
        ),
        doc="An array of wavelengths specifying the translation wavelength of each "
        "Dirac delta. Wavelength values are positive and unique."
        "When a single value is provided, it is converted to a 1-element array."
        "Wavelength values are sorted by increasing magnitude.",
        type="quantity",
        init_type="array-like or quantity",
    )

    @wavelengths.validator
    def _w_validator(self, attribute, value):
        if not np.all(value > 0):
            raise ValueError(f"w values must be all positive (got {value})")
        if np.unique(value.m).size != value.m.size:
            raise ValueError(f"w values must be unique (got {value})")

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        raise NotImplementedError

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError

    @property
    def template(self) -> dict:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        raise NotImplementedError

    def select_in_wavelength_set(self, wset: WavelengthSet) -> WavelengthSet:
        # the input wavelength set is completely ignored
        # only the attribute wavelengths are included in the returned
        # wavelength set

        return WavelengthSet(self.wavelengths)

    def select_in_bin_set(self, binset: BinSet) -> BinSet:
        bins = binset.bins
        wunits = "nm"
        xmin = np.array([bin.wmin.m_as(wunits) for bin in bins])
        xmax = np.array([bin.wmax.m_as(wunits) for bin in bins])
        x = self.wavelengths.m_as(wunits)
        selected = _select(xmin, xmax, x)
        return BinSet(bins=list(np.array(bins)[selected]))


def _select(xmin, xmax, x):
    selmin = np.searchsorted(xmin, x)
    selmax = np.searchsorted(xmax, x) + 1
    hit = selmin == selmax  # Mask where x values which triggered a bin hit

    # Map x values to selected bin (index -999 means not selected)
    bin_index = np.where(hit, selmin - 1, np.full_like(x, -999)).astype("int")

    # Get selected bins only
    selected = np.unique(bin_index)[bin_index >= 0]  # mask removes -999 value

    return selected
