import typing as t

import attrs
import mitsuba as mi

from ._core import BSDF
from ..core import NodeSceneElement, Param, ParamFlags
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import documented, parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class CheckerboardBSDF(BSDF, NodeSceneElement):
    """
    Checkerboard BSDF [``checkerboard``].

    This class defines a Lambertian BSDF textured with a checkerboard pattern.
    """

    reflectance_a: Spectrum = documented(
        attrs.field(
            default=0.2,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.2",
    )

    reflectance_b: Spectrum = documented(
        attrs.field(
            default=0.8,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=":class:`.Spectrum` or dict or float",
        default="0.8",
    )

    scale_pattern: float = documented(
        attrs.field(
            default=2.0, converter=float, validator=attrs.validators.instance_of(float)
        ),
        doc="Scaling factor for the checkerboard pattern. The higher the value, "
        "the more checkboard patterns will fit on the surface to which this "
        "reflection model is attached.",
        type="float",
        default=2.0,
    )

    @property
    def template(self) -> dict:
        return {
            "type": "diffuse",
            "reflectance": {"type": "checkerboard"},
        }

    @property
    def params(self) -> t.Dict[str, Param]:
        return {
            "reflectance.color0": Param(
                lambda ctx: self.reflectance_a.eval(ctx.spectral_ctx),
                ParamFlags.SPECTRAL,
            ),
            "reflectance.color1": Param(
                lambda ctx: self.reflectance_a.eval(ctx.spectral_ctx),
                ParamFlags.SPECTRAL,
            ),
            "reflectance.to_uv": Param(
                lambda ctx: mi.ScalarTransform4f.scale(self.scale_pattern)
            ),
        }
