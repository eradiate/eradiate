import attr

from ..core import SceneElement
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory


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


class IntegratorFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`Integrator`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: IntegratorFactory
    """

    _constructed_type = Integrator
    registry = {}
