from __future__ import annotations

from abc import ABC

import attrs

from ..core import NodeSceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

illumination_factory = Factory()
illumination_factory.register_lazy_batch(
    [
        ("_astro_object.AstroObjectIllumination", "astro_object", {}),
        ("_constant.ConstantIllumination", "constant", {}),
        ("_directional.DirectionalIllumination", "directional", {}),
        ("_spot.SpotIllumination", "spot", {}),
    ],
    cls_prefix="eradiate.scenes.illumination",
)


@parse_docs
@attrs.define(eq=False, slots=False)
class Illumination(NodeSceneElement, ABC):
    """
    Abstract base class for all illumination scene elements.

    Notes
    -----
    * This class is to be used as a mixin.
    """

    id: str | None = documented(
        attrs.field(
            default="illumination",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"illumination"',
    )
