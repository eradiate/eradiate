from abc import ABC

import attr

from ..core import SceneElement
from ..._factory import BaseFactory


@attr.s
class Illumination(SceneElement, ABC):
    """Abstract base class for all illumination scene elements.

    See :class:`~eradiate.scenes.core.SceneElement` for undocumented members.
    """

    id = attr.ib(
        default="illumination",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )


class IlluminationFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`Illumination`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: IlluminationFactory
    """

    _constructed_type = Illumination
    registry = {}
