from typing import MutableMapping, Optional

import attr

from ..core import SceneElement
from ..._factory import BaseFactory
from ...attrs import parse_docs
from ...contexts import KernelDictContext


@parse_docs
@attr.s
class CanopyElement(SceneElement):
    """
    Abstract base class for objects that can be instantiated in an
    :class:`.InstancedCanopyElement`.
    """

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if not ctx.ref:
            return self.shapes(ctx=ctx)
        else:
            return {**self.bsdfs(ctx=ctx), **self.shapes(ctx=ctx)}


class CanopyElementFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`.CanopyElement`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: CanopyElementFactory
    """

    _constructed_type = CanopyElement
    registry = {}
