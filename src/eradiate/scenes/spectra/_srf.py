import attrs
import numpy as np
import pint
import pinttr

import eradiate

from ._core import Spectrum
from ..core import KernelDict
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext
from ...exceptions import UnsupportedModeError
from ...units import PhysicalQuantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@parse_docs
@attrs.define
class RectangularSRF(Spectrum):
    """
    A rectangular spectral response function [``rectangular_srf``].

    Notes
    -----
    The response function is defined as:
    
    .. math::
        R(\\lambda) = \\begin{cases}
            1 & | \\lambda - \\lambda_0 | <= w/2 \\\\
            0 & \\text{otherwise}
        \\end{cases}

    where:
    
    * :math:`\\lambda_0` is set by the ``wavelength`` parameter.
    * :math:`w` is set by the ``width`` parameter.

    In ``mono_*`` modes, :math:`\\lambda` is the wavelength associated to the
    current spectral context.
    In ``ckd_*`` modes, :math:`\\lambda` is the center wavelength of the 
    spectral bin associated to the current spectral context.
    """
    quantity: PhysicalQuantity = attrs.field(
        default=PhysicalQuantity.DIMENSIONLESS, init=False, repr=False
    )

    wavelength: pint.Quantity = documented(
        pinttr.field(
            default=550.0 * ureg.nm,
            units=ucc.deferred("wavelength"),
            kw_only=True,
        ),
        doc="Center wavelength.",
        type="quantity",
    )

    width: pint.Quantity = documented(
        pinttr.ib(
            default=10.0 * ureg.nm,
            units=ucc.deferred("wavelength"),
            kw_only=True,
        ),
        doc="Wavelength width.",
        type="quantity",
    )

    @property
    def start(self) -> pint.Quantity:
        """Start wavelength."""
        return self.wavelength - self.width / 2.0
    
    @property
    def stop(self) -> pint.Quantity:
        """Stop wavelength."""
        return self.wavelength + self.width / 2.0

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        wunits = self.wavelength.units
        in_rectangle = np.isclose(
            w.m_as(wunits),
            self.wavelength.m_as(wunits),
            atol=self.width.m_as(wunits) / 2.0,
        )
        return np.where(in_rectangle, 1.0, 0.0) * ureg.dimensionless

    def eval_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        result = np.zeros((len(bindexes),))
        for i_bindex, bindex in enumerate(bindexes):
            bin = bindex.bin
            result[i_bindex] = self.eval_mono(bin.wcenter)
            
        return result

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        if wmax < self.start or wmin > self.stop:
            return 0.0 * ureg.dimensionless
        else:
            if wmin < self.start:
                wmin = self.start
            if wmax > self.stop:
                wmax = self.stop
            return (wmax - wmin) * (1.0 * ureg.dimensionless)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
            value = float(self.eval(ctx.spectral_ctx).m_as(uck.get(self.quantity)))
            return KernelDict({"spectrum": {"type": "uniform", "value": value}})
        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))
        
