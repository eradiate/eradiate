import typing as t
from abc import ABC

import attr

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

phase_function_factory = Factory()


@parse_docs
@attr.s
class PhaseFunction(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all phase functions.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="phase",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"phase"',
    )
