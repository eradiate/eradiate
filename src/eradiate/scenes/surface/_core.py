from __future__ import annotations

import typing as t
from abc import ABC

import attrs

from ..core import CompositeSceneElement, NodeSceneElement
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
@attrs.define(eq=False, slots=False)
class Surface:
    """
    An abstract base class defining common facilities for all surfaces.

    All scene elements deriving from this interface are composite and cannot be
    turned into a kernel scene instance on their own: they must be owned by a
    container which take care of expanding them.

    Notes
    -----
    * This class is to be used as a mixin.
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default="surface",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"surface"',
    )


@attrs.define(eq=False, slots=False)
class SurfaceComposite(Surface, CompositeSceneElement, ABC):
    pass
