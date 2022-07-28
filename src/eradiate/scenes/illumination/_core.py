import typing as t
from abc import ABC

import attr

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

illumination_factory = Factory()
illumination_factory.register_lazy_batch(
    [
        ("_constant.ConstantIllumination", "constant", {}),
        ("_directional.DirectionalIllumination", "directional", {}),
    ],
    cls_prefix="eradiate.scenes.illumination",
)


@parse_docs
@attr.s
class Illumination(SceneElement, ABC):
    """
    Abstract base class for all illumination scene elements.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="illumination",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"illumination"',
    )
