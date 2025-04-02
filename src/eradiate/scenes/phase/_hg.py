from __future__ import annotations

import attrs

from ._core import PhaseFunction
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter


@define(eq=False, slots=False)
class HenyeyGreensteinPhaseFunction(PhaseFunction):
    """
    Henyey-Greenstein phase function [``hg``].

    The Henyey-Greenstein phase function :cite:`Henyey1941Diffuse` models
    scattering in an isotropic medium. The scattering pattern is controlled by
    its :math:`g` parameter, which is equal to the phase function's asymmetry
    parameter (the mean cosine of the scattering angle): a positive (resp.
    negative) value corresponds to predominant forward (resp. backward)
    scattering.
    """

    g: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Asymmetry parameter. Must be dimensionless. Must be in :math:`]-1, 1[`.",
        type=":class:`.Spectrum`",
        init_type=":class:`.Spectrum` or dict or float, optional",
        default="0.0",
    )

    @property
    def template(self) -> dict:
        return {
            "type": "hg",
            "g": DictParameter(lambda ctx: float(self.g.eval(ctx.si))),
        }

    @property
    def params(self) -> dict[str, SceneParameter]:
        return {
            "g": SceneParameter(
                lambda ctx: float(self.g.eval(ctx.si)),
                KernelSceneParameterFlags.SPECTRAL,
            )
        }
