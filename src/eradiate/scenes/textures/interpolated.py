from __future__ import annotations

from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import pinttr

from ..core import NodeSceneElement
from ...attrs import documented, parse_docs
from ...kernel import InitParameter, UpdateParameter
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import unit_context_config as ucc


# the "data" parameter must be first in the args list of the function provided to
# `np.apply_along_axis` but third in the args of `np.interp`
def my_interp1d(data, w, wavelengths):
    return np.interp(w, wavelengths, data, left=0.0, right=0.0)


@parse_docs
@attrs.define(eq=False, slots=False)
class InterpolatedTexture(NodeSceneElement):
    """
    Spectrally interpolated texture.

    Notes
    -----
    Interpolation uses :func:`numpy.interp`. Evaluation is as follows:

    * in ``mono_*`` modes, the spectrum is evaluated at the spectral context
      wavelength;
    * in ``ckd_*`` modes, the spectrum is evaluated as the average value over
      the spectral context bin.
    """

    wavelengths: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("wavelength"),
            converter=[
                np.atleast_1d,
                pinttr.converters.to_units(ucc.deferred("wavelength")),
            ],
            kw_only=True,
        ),
        doc="Wavelengths defining the interpolation grid. Values must be "
        "monotonically increasing.",
        type="quantity",
    )

    @wavelengths.validator
    def _wavelengths_validator(self, attribute, value):
        # wavelength must be monotonically increasing
        if not np.all(np.diff(value) > 0):
            raise ValueError("wavelengths must be monotonically increasing")

    data: np.typing.ArrayLike = documented(
        attrs.field(
            converter=np.atleast_3d,
            kw_only=True,
        ),
        doc="Texture data. Must be 3d and the third dimension must match "
        "the length of ``wavelengths``",
        type="array-like",
        init_type="array-like",
    )

    @data.validator
    def _data_validator(self, attribute, value):
        if not np.shape(value)[2] == np.shape(self.wavelengths)[0]:
            raise ValueError(
                f"while validating '{attribute.name}': Spectral dimension must match"
                f"length of wavelength array."
            )
        if not (value.shape[0] >= 2 and value.shape[1] <= 2):
            raise ValueError(
                f"while validating '{attribute.name}': Texture must at least have"
                f"2 by 2 pixels."
            )

    @singledispatchmethod
    def eval(self, si: SpectralIndex) -> pint.Quantity:
        """
        Evaluate spectrum at a given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        value : quantity
            Evaluated spectrum.

        Notes
        -----
        This method dispatches evaluation to specialized methods depending
        on the spectral index type.
        """
        raise NotImplementedError

    @eval.register(MonoSpectralIndex)
    def _(self, si) -> pint.Quantity:
        return self.eval_mono(w=si.w)

    @eval.register(CKDSpectralIndex)
    def _(self, si) -> pint.Quantity:
        return self.eval_ckd(w=si.w, g=si.g)

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return np.atleast_3d(
            np.apply_along_axis(my_interp1d, 2, self.data, w, self.wavelengths)
        )

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_mono(w=w)

    @property
    def template(self) -> dict:
        # Inherit docstring

        return {
            "type": "bitmap",
            "filter_type": "nearest",
            "wrap_mode": "clamp",
            "raw": True,
            "data": InitParameter(evaluator=lambda ctx: self.eval(ctx.si)),
        }

    @property
    def params(self) -> dict:
        # Inherit docstring

        return {
            "data": UpdateParameter(
                evaluator=lambda ctx: self.eval(ctx.si),
                flags=UpdateParameter.Flags.SPECTRAL,
            )
        }
