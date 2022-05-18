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


@phase_function_factory.register(type_id="tab_phase")
@parse_docs
@attr.s
class TabulatedPhaseFunction(PhaseFunction):
    r"""
    Tabulated phase function [``tab_phase``].

    A lookup table-based phase function. The ``data`` field is a
    :class:`~xarray.DataArray` with wavelength and angular dimensions.

    Notes
    -----
    The :math:`\mu` coordinate must cover the :math:`[-1, 1]` interval but
    there is no constraints on the ordering of values nor on the spacing
    between two consecutive values, i.e. non-regular :math:`\mu` grid are
    supported. If the :math:`\mu` grid is non-regular and since the
    underlying mitsuba ``tabphase`` plugin expects phase function values
    on a regular :math:`\mu` grid, we compute a regular grid with a
    :math:`\mu` step equal to the small :math:`\mu` step found in the input
    `data` and interpolate the latter on that regular grid. This can
    lead to large arrays (several 100 MB). If you want to control
    the size of the :math:`\mu` grid yourself, you can simply provide a
    :class:`~xarray.DataArray` object that already has a regular grid
    :math:`\mu` coordinate ; in such a case, the :class:`~xarray.DataArray`
    object is not further interpolated along the :math:`\mu` dimension.
    For better performance, it is best to provide phase function data on a
    regular (and sorted) :math:`\mu` grid, since neither conversion nor
    interpolation is performed in that case.
    """

    data: xr.DataArray = documented(
        attr.ib(
            converter=_convert_data,
            validator=[attr.validators.instance_of(xr.DataArray), _validate_data],
            kw_only=True,
        ),
        type="DataArray",
        doc="Value table as a data array with wavelength (``w``), scattering "
        "angle cosine (``mu``), and scattering phase matrix row "
        "(``i``) and column (``j``) indices (integer) as coordinates. "
        "This parameter has no default.",
    )

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
        w_units = self.data.w.attrs["units"]

        # data already maps to regular grid of scattering angle cosines
        dmu = self.data.mu.values[1:] - self.data.mu.values[:-1]
        if np.allclose(dmu, dmu[0], rtol=1e-8):
            return (
                self.data.sel(i=0, j=0)
                .interp(
                    w=w.m_as(w_units),
                    kwargs=dict(bounds_error=True),
                )
                .data
            )

        # data does not map to regular grid of scattering angle cosines
        else:
            # compute the regular grid from the smallest mu step found in the
            # input data
            dmu_min = np.abs(self.data.mu.diff(dim="mu")).values.min()
            nmu = int(np.ceil(2.0 / dmu_min)) + 1
            mu = np.linspace(-1.0, 1.0, nmu)

            # interpolate first on wavelength
            _w = _ensure_magnitude_array(w)
            data_w_interpolated = self.data.sel(i=0, j=0).interp(
                w=_w.m_as(w_units),
                kwargs=dict(bounds_error=True),
            )

            # for performance, interpolation on mu dimension is performed with
            # numpy.interp, which is found to be more than twice faster than
            # xarray.DataArray.interp
            phase = np.full(shape=(_w.size, nmu), fill_value=np.nan)
            mup = self.data.mu.values
            for iw in range(_w.size):
                fp = data_w_interpolated.isel(w=iw).values
                phase[iw, :] = np.interp(x=mu, xp=mup, fp=fp)

            return phase.squeeze()

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
