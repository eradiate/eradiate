import typing as t

import attrs

from ._core import BSDFNode
from ..core import NodeSceneElement
from ..spectra import SpectrumNode, spectrum_factory
from ... import validators
from ...attrs import documented, parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class CheckerboardBSDF(BSDFNode):
    """
    Checkerboard BSDF [``checkerboard``].

    This class defines a Lambertian BSDF textured with a checkerboard pattern.
    """

    reflectance_a: SpectrumNode = documented(
        attrs.field(
            default=0.2,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(SpectrumNode),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".SpectrumNode",
        init_type=".SpectrumNode or dict or float",
        default="0.2",
    )

    reflectance_b: SpectrumNode = documented(
        attrs.field(
            default=0.8,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(SpectrumNode),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".SpectrumNode",
        init_type=":class:`.SpectrumNode` or dict or float",
        default="0.8",
    )

    scale_pattern: float = documented(
        attrs.field(
            default=2.0, converter=float, validator=attrs.validators.instance_of(float)
        ),
        doc="Scaling factor for the checkerboard pattern. The higher the value, "
        "the more checkerboard patterns will fit on the surface to which this "
        "reflection model is attached.",
        type="float",
        default="2.0",
    )

    @property
    def template(self) -> dict:
        return {
            "type": "diffuse",
            "reflectance.type": "checkerboard",
        }

    @property
    def objects(self) -> t.Dict[str, NodeSceneElement]:
        return {
            "reflectance.color0": self.reflectance_a,
            "reflectance.color1": self.reflectance_b,
        }
