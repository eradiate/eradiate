import attrs
import numpy as np
import pint
import pinttr

from ._core import Spectrum
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@parse_docs
@attrs.define
class MultiDeltaSpectrum(Spectrum):
    """
    A spectrum made of multiple translated Dirac delta [``multi_delta``].

    Attributes
    ----------
    wavelengths : pint.Quantity
        An array of wavelengths specifying the translation wavelength of each
        Dirac delta. Wavelength values are positive and unique.
    
    Parameters
    ----------
    w : pint.Quantity
        An array of wavelengths specifying the translation wavelength of each
        Dirac delta.
        When a single value is provided, it is converted to a 1-element array.

    Notes
    -----
    This spectrum is for special use.
    It does not have a kernel representation.
    The spectrum cannot be evaluated.
    As a result, neither can it be plotted.
    """
    wavelengths : pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=uck.deferred("wavelength"),
            converter=np.atleast_1d,
        ),
        doc="An array of wavelengths specifying the translation wavelength of "
            "each Dirac delta.",
        type="quantity",
    )

    @wavelengths.validator
    def _w_validator(self, attribute, value):
        if not np.all(value > 0):
            raise ValueError(f"w values must be all positive (got {value})")
        if np.unique(value.m).size != value.m.size:
            raise ValueError(f"w values must be unique (got {value})")

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        raise NotImplementedError
    
    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError
    
    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        raise NotImplementedError
    
    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError
