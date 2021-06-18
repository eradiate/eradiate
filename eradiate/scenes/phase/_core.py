from abc import ABC
from typing import Optional

import attr

from ..core import SceneElement
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory


@parse_docs
@attr.s
class PhaseFunction(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all phase functions.
    """

    id: Optional[str] = documented(
        attr.ib(
            default="phase",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"phase"',
    )


class PhaseFunctionFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`PhaseFunction`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: PhaseFunction
    """

    _constructed_type = PhaseFunction
    registry = {}
