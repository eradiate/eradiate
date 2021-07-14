from abc import ABC

import attr

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

illumination_factory = Factory()


@parse_docs
@attr.s
class Illumination(SceneElement, ABC):
    """
    Abstract base class for all illumination scene elements.
    """

    id = documented(
        attr.ib(
            default="illumination",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"illumination"',
    )
