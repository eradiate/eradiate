from __future__ import annotations

from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import xarray as xr

from ._core import PhaseFunction
from ...attrs import documented, parse_docs
from ...kernel import InitParameter, UpdateParameter
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import unit_registry as ureg
from ...util.misc import summary_repr


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
class TabulatedPhaseFunction(PhaseFunction):
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
            repr=summary_repr,
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

    @singledispatchmethod
    def eval(self, si: SpectralIndex) -> np.ndarray:
        """
        Evaluate phase function at a given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D array.

        Notes
        -----
        This method dispatches evaluation to specialized methods depending on
        the spectral index type.
        """
        raise NotImplementedError

    @eval.register(MonoSpectralIndex)
    def _(self, si) -> np.ndarray:
        return self.eval_mono(w=si.w)

    @eval.register(CKDSpectralIndex)
    def _(self, si) -> np.ndarray:
        return self.eval_ckd(w=si.w, g=si.g)

    def eval_mono(self, w: pint.Quantity) -> np.ndarray:
        """
        Evaluate phase function in momochromatic modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Wavelength.

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

    def eval_ckd(self, w: pint.Quantity, g: float) -> np.ndarray:
        """
        Evaluate phase function in ckd modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Spectral bin center wavelength.

        g : float
            Absorption coefficient cumulative probability.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the shape
            of `w` (angle dimension comes last).
        """
        return self.eval_mono(w=w)

    @property
    def template(self):
        result = {
            "type": "tabphase" if not self._is_irregular else "tabphase_irregular",
            "values": InitParameter(
                lambda ctx: ",".join(map(str, self.eval(ctx.si))),
            ),
        }

        if self._is_irregular:
            result["nodes"] = InitParameter(
                lambda ctx: ",".join(map(str, self.data.mu.values))
            )

        return result

    @property
    def params(self) -> dict[str, UpdateParameter]:
        return {
            "values": UpdateParameter(
                lambda ctx: self.eval(ctx.si),
                UpdateParameter.Flags.SPECTRAL,
            )
        }
