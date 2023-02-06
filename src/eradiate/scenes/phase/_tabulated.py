import typing as t

import attrs
import numpy as np
import pint
import xarray as xr

import eradiate

from ._core import PhaseFunctionNode
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import SpectralContext
from ...exceptions import UnsupportedModeError
from ...kernel import InitParameter, UpdateParameter
from ...units import unit_registry as ureg


def _ensure_magnitude_array(q: pint.Quantity) -> pint.Quantity:
    """Convert to Quantity whose magnitude is an array"""
    if np.isscalar(q.magnitude):
        return ureg.Quantity(np.atleast_1d(q.magnitude), q.units)
    else:
        return q


def _convert_data(da: xr.DataArray) -> xr.DataArray:
    """
    If the coordinate mu is not strictly increasing, return a reindexed
    copy of the DataArray where it is.
    Else, return the input DataArray.
    """
    if np.all(da["mu"].values[1:] > da["mu"].mu.values[:-1]):
        return da
    elif np.all(da["mu"].values[1:] < da["mu"].mu.values[:-1]):
        return da.reindex(mu=da.mu[::-1])
    else:
        return da.reindex(mu=np.sort(da.mu.values))


def _validate_data(instance, attribute, value):
    """
    Bounds of mu coordinate must be [-1.0, 1.0].
    """
    mu_min = value["mu"].values.min()
    mu_max = value["mu"].values.max()

    if not (mu_min == -1.0 and mu_max == 1.0):
        raise ValueError(
            f"Coordinate 'mu' bounds must be -1.0, 1.0 (got {mu_min}, {mu_max})."
        )


@parse_docs
@attrs.define(eq=False, slots=False)
class TabulatedPhaseFunction(PhaseFunctionNode):
    r"""
    Tabulated phase function [``tab_phase``].

    A lookup table-based phase function. The ``data`` field is a
    :class:`~xarray.DataArray` with wavelength and angular dimensions.

    Notes
    -----
    * The :math:`\mu` coordinate must cover the :math:`[-1, 1]` interval but
      there is no constraint on value ordering or spacing. In particular,
      irregular :math:`\mu` grids are supported.

    * For optimal performance, providing phase function data on a regular,
      sorted :math:`\mu` grid is recommended.
    """

    data: xr.DataArray = documented(
        attrs.field(
            converter=_convert_data,
            validator=[attrs.validators.instance_of(xr.DataArray), _validate_data],
            kw_only=True,
        ),
        type="DataArray",
        doc="Value table as a data array with wavelength (``w``), scattering "
        "angle cosine (``mu``), and scattering phase matrix row "
        "(``i``) and column (``j``) indices (integer) as coordinates. "
        "This parameter has no default.",
    )

    _is_irregular: bool = attrs.field(default=False, init=False, repr=False)

    def __attrs_post_init__(self):
        # Check whether mu coordinate spacing is regular
        mu = self.data.mu.values
        dmu = mu[1:] - mu[:-1]
        self._is_irregular = not np.allclose(dmu, dmu[0])

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
        w_units = self.data.w.attrs["units"]

        return (
            self.data.isel(i=0, j=0)
            .interp(
                w=w.m_as(w_units),
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

    @property
    def template(self):
        result = {
            "type": "tabphase" if not self._is_irregular else "tabphase_irregular",
            "values": InitParameter(
                lambda ctx: ",".join(
                    map(str, self.eval(spectral_ctx=ctx.spectral_ctx))
                ),
            ),
        }

        if self._is_irregular:
            result["nodes"] = InitParameter(
                lambda ctx: ",".join(map(str, self.data.mu.values))
            )

        return result

    @property
    def params(self) -> t.Dict[str, UpdateParameter]:
        return {
            "values": UpdateParameter(
                lambda ctx: self.eval(spectral_ctx=ctx.spectral_ctx),
                UpdateParameter.Flags.SPECTRAL,
            )
        }
