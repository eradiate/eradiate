import attr

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

integrator_factory = Factory()


@parse_docs
@attr.s
class Integrator(SceneElement):
    """
    Abstract base class for all integrator elements.
    """

    id: str = documented(
        attr.ib(
            default="integrator",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"integrator"',
    )
