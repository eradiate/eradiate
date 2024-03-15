from __future__ import annotations

import attrs
import numpy as np
import pint
import pinttr

from ._core import Spectrum
from ... import validators
from ...attrs import documented, parse_docs
from ...spectral.ckd import BinSet
from ...spectral.mono import WavelengthSet
from ...units import unit_context_kernel as uck
from ...util.misc import summary_repr_vector


@parse_docs
@attrs.define(eq=False, slots=False)
class MultiDeltaSpectrum(Spectrum):
    """
    A spectrum made of multiple translated Dirac delta [``multi_delta``].

    Notes
    -----
    This spectrum is intended to be used for spectral grid specification.
    It has no kernel-level representation, and it cannot be evaluated.
    """

    wavelengths: pint.Quantity = documented(
        pinttr.field(
            kw_only=True,
            units=uck.deferred("wavelength"),
            converter=[
                lambda x: np.sort(np.atleast_1d(x)),
                pinttr.converters.to_units(uck.deferred("wavelength")),
                lambda x: np.unique(x.m) * x.units,
            ],
            validator=validators.all_strictly_positive,
            repr=lambda x: f"{summary_repr_vector(x.m)} {x.u:~}",
        ),
        doc="An array of wavelengths specifying the translation wavelength of each "
        "Dirac delta. Wavelength values are positive and unique. "
        "When a single value is provided, it is converted to a 1-element array. "
        "Wavelength are deduplicated and sorted by ascending values. "
        'Unit-enabled field (default: ``ucc["wavelength"]``).',
        type="quantity",
        init_type="array-like or quantity",
    )

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
    bin_index = np.where(hit, selmin - 1, np.full_like(x, -999)).astype(np.int64)

    # Get selected bins only
    selected = np.unique(bin_index)  # mask removes -999 value
    selected = selected[selected >= 0]

    return selected
