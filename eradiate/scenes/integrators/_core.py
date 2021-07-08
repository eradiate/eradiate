import attr
from dessinemoi import Factory

from ..core import SceneElement
from ...attrs import documented, get_doc, parse_docs


@parse_docs
@attr.s
class Integrator(SceneElement):
    """
    Abstract base class for all integrator elements.
    """

    id = documented(
        attr.ib(
            default="integrator",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"integrator"',
    )


integrator_factory = Factory()
