from abc import ABC
from typing import Optional

import attr
from dessinemoi import Factory

from ..core import SceneElement
from ...attrs import documented, get_doc, parse_docs


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


phase_function_factory = Factory()
