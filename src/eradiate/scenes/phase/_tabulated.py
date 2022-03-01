import attr
import numpy as np
import pint
import xarray as xr

import eradiate

from ._core import PhaseFunction, phase_function_factory
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...units import unit_registry as ureg


@phase_function_factory.register(type_id="tab_phase")
@parse_docs
@attr.s
class TabulatedPhaseFunction(PhaseFunction):
    """
    Tabulated phase function [``tab_phase``].

    A lookup table-based phase function. The `data` field is a
    :class:`~xarray.DataArray` with wavelength and angular dimensions.
    """

    data: xr.DataArray = documented(
        attr.ib(
            validator=attr.validators.instance_of(xr.DataArray),
            kw_only=True,
        ),
        type="DataArray",
        doc="Value table as a data array with wavelength (``w``) and "
        "scattering angle cosine (``mu``) as coordinates. This parameter has "
        "no default.",
    )

    # Number of points used to represent the phase function on the angular coordinate
    _n_mu: int = attr.ib(default=201, init=False, repr=False)

    def eval(self, spectral_ctx: SpectralContext) -> np.ndarray:
        r"""
        Evaluate phase function based on a spectral context. This method
        dispatches evaluation to specialised methods depending on the active
        mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D array.

        Notes
        -----
        The phase function is represented by an array of values mapped to
        regularly spaced scattering angle cosine values
        (:math:`\mu \in [-1, 1]`).
        """
        if eradiate.mode().is_mono:
            return self.eval_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_mono(self, w: pint.Quantity) -> np.ndarray:
        """
        Evaluate phase function in monochromatic modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Wavelength values at which the phase function is to be evaluated.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the shape
            of `w` (angle dimension comes last).
        """
        return (
            self.data.sel(i=0, j=0)
            .interp(
                w=w.m_as(self.data.w.units),
                mu=np.linspace(-1, 1, self._n_mu),
                kwargs=dict(bounds_error=True),
            )
            .data
        )

    def eval_ckd(self, *bindexes: Bindex) -> np.ndarray:
        """
        Evaluate phase function in CKD modes.

        Parameters
        ----------
        *bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the phase
            function.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the number
            of passed bindexes (angle dimension comes last).
        """
        w_units = ureg(self.data.w.units)
        w = [bindex.bin.wcenter.m_as(w_units) for bindex in bindexes] * w_units
        return self.eval_mono(w)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        return KernelDict(
            {
                self.id: {
                    "type": "tabphase",
                    "values": ",".join(
                        map(str, [value for value in self.eval(ctx.spectral_ctx)])
                    ),
                }
            }
        )
