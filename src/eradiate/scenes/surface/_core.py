from __future__ import annotations

import typing as t
from abc import ABC

import attrs

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

surface_factory = Factory()
surface_factory.register_lazy_batch(
    [
        ("_basic.BasicSurface", "basic", {}),
        ("_central_patch.CentralPatchSurface", "central_patch", {}),
        ("_dem.DEMSurface", "dem", {}),
    ],
    cls_prefix="eradiate.scenes.surface",
)


@parse_docs
@attrs.define(eq=False)
class Surface(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all surfaces.

    All scene elements deriving from this interface are composite and cannot be
    turned into a kernel scene instance on their own: they must be owned by a
    container which take care of expanding them.
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default="surface",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"surface"',
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        """
        Enforce internal state consistency. This method should be called when
        fields are modified. It is automatically called as a post-init step.
        The default implementation is a no-op.
        """
        pass

    @property
    def kernel_type(self) -> None:
        return None
