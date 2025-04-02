from __future__ import annotations

from functools import singledispatchmethod
from typing import Literal

import attrs
import numpy as np
import pint
import xarray as xr

import eradiate

from ._core import PhaseFunction
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
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


@define(eq=False, slots=False)
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

    force_polarized_phase: bool = documented(
        attrs.field(default=False, converter=bool, kw_only=True),
        doc="Flag that forces the use of a polarized phase function.",
        type="bool",
        init_type="bool",
        default="False",
    )

    particle_shape: Literal["spherical", "spheroidal"] = documented(
        attrs.field(default="spherical", kw_only=True),
        doc="Defines the shape of the particle. Only used in polarized mode.\n\n"
        '* ``"spherical"``: 4 coefficients considered [m11, m12, m33, m34].\n'
        '* ``"spheroidal"``: 6 coefficients considered [m11, m12, m22, m33, m34, m44].',
        type="str",
        init_type='{"spherical", "spheroidal"}',
        default='"spherical"',
    )

    _is_irregular: bool = attrs.field(default=False, init=False, repr=False)

    @property
    def has_polarized_data(self) -> bool:
        return self.data.i.shape[0] == 4 or self.data.j.shape[0] == 4

    @property
    def is_polarized(self) -> bool:
        return eradiate.mode().is_polarized and (
            self.has_polarized_data or self.force_polarized_phase
        )

    def __attrs_post_init__(self):
        # Check whether mu coordinate spacing is regular
        mu = self.data.mu.values
        dmu = mu[1:] - mu[:-1]

        self._is_irregular = not np.allclose(dmu, dmu[0]) or self.is_polarized

    @singledispatchmethod
    def eval(self, si: SpectralIndex, i: int, j: int) -> np.ndarray:
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
    def _(self, si, i, j) -> np.ndarray:
        return self.eval_mono(w=si.w, i=i, j=j)

    @eval.register(CKDSpectralIndex)
    def _(self, si, i, j) -> np.ndarray:
        return self.eval_ckd(w=si.w, g=si.g, i=i, j=j)

    def eval_mono(self, w: pint.Quantity, i, j) -> np.ndarray:
        """
        Evaluate phase function in momochromatic modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Wavelength.

        i : int
            Phase matrix row index

        j : int
            Phase matrix column index

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the shape
            of `w` (angle dimension comes last).
        """
        w_units = self.data.w.attrs["units"]

        return (
            self.data.isel(i=i, j=j)
            .interp(
                w=w.m_as(w_units),
                kwargs=dict(bounds_error=True),
            )
            .data
        )

    def eval_ckd(self, w: pint.Quantity, g: float, i: int, j: int) -> np.ndarray:
        """
        Evaluate phase function in ckd modes.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Spectral bin center wavelength.

        i : int
            Phase matrix row index

        j : int
            Phase matrix column index

        g : float
            Absorption coefficient cumulative probability.

        Returns
        -------
        ndarray
            Evaluated phase function as a 1D or 2D array depending on the shape
            of `w` (angle dimension comes last).
        """
        return self.eval_mono(w=w, i=i, j=j)

    @property
    def template(self):
        phase_function = "tabphase"
        values_name = "values"

        if self.is_polarized:
            phase_function = "tabphase_polarized"
            values_name = "m11"
        else:
            if self._is_irregular:
                phase_function = "tabphase_irregular"

        result = {
            "type": phase_function,
            values_name: DictParameter(
                lambda ctx: ",".join(map(str, self.eval(ctx.si, 0, 0))),
            ),
        }

        if self.is_polarized:
            if self.has_polarized_data:
                result["m12"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, 0, 1))),
                )
                result["m33"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, 2, 2))),
                )
                result["m34"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, 2, 3))),
                )

                if self.particle_shape == "spheroidal":
                    result["m22"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, 1, 1))),
                    )

                    result["m44"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, 3, 3))),
                    )

                elif self.particle_shape == "spherical":
                    result["m22"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, 0, 0))),
                    )

                    result["m44"] = DictParameter(
                        lambda ctx: ",".join(map(str, self.eval(ctx.si, 2, 2))),
                    )

                else:
                    raise NotImplementedError

            else:
                # case: no polarized data but forced polarized. Initialize the
                # diagonal to have the same behaviour as with tabphase in
                # polarized mode.
                result["m22"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, 0, 0))),
                )

                result["m33"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, 0, 0))),
                )

                result["m44"] = DictParameter(
                    lambda ctx: ",".join(map(str, self.eval(ctx.si, 0, 0))),
                )

        if self._is_irregular:
            result["nodes"] = DictParameter(
                lambda ctx: ",".join(map(str, self.data.mu.values))
            )

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        values_name = "values"

        if self.is_polarized:
            values_name = "m11"

        result = {
            values_name: SceneParameter(
                lambda ctx: self.eval(ctx.si, 0, 0),
                KernelSceneParameterFlags.SPECTRAL,
            )
        }

        if self.is_polarized:
            if self.has_polarized_data:
                result["m12"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 1),
                    KernelSceneParameterFlags.SPECTRAL,
                )
                result["m33"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 2, 2),
                    KernelSceneParameterFlags.SPECTRAL,
                )
                result["m34"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 2, 3),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                if self.particle_shape == "spheroidal":
                    result["m22"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 1, 1),
                        KernelSceneParameterFlags.SPECTRAL,
                    )
                    result["m44"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 3, 3),
                        KernelSceneParameterFlags.SPECTRAL,
                    )

                elif self.particle_shape == "spherical":
                    result["m22"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 0, 0),
                        KernelSceneParameterFlags.SPECTRAL,
                    )
                    result["m44"] = SceneParameter(
                        lambda ctx: self.eval(ctx.si, 2, 2),
                        KernelSceneParameterFlags.SPECTRAL,
                    )

                else:
                    raise NotImplementedError

            else:
                # case: no polarized data but forced polarized. Initialize the
                # diagonal to have the same behaviour as with tabphase in
                # polarized mode.
                result["m22"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 0),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                result["m33"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 0),
                    KernelSceneParameterFlags.SPECTRAL,
                )

                result["m44"] = SceneParameter(
                    lambda ctx: self.eval(ctx.si, 0, 0),
                    KernelSceneParameterFlags.SPECTRAL,
                )

        return result
