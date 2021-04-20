from abc import ABC

import attr

from ..core import SceneElement
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory


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


class IlluminationFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`Illumination`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: IlluminationFactory
    """

    _constructed_type = Illumination
    registry = {}
