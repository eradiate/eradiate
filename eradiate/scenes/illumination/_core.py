from abc import ABC

import attr
from dessinemoi import Factory

from ..core import SceneElement
from ...attrs import documented, get_doc, parse_docs


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


illumination_factory = Factory()
