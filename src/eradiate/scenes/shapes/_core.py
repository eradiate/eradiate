from __future__ import annotations

import typing as t
from abc import ABC

import attrs

from ..bsdfs import BSDF, bsdf_factory
from ..core import InstanceSceneElement, NodeSceneElement, Ref
from ..._factory import Factory
from ...attrs import documented, get_doc, parse_docs

shape_factory = Factory()
shape_factory.register_lazy_batch(
    [
        ("_cuboid.CuboidShape", "cuboid", {}),
        ("_rectangle.RectangleShape", "rectangle", {}),
        ("_sphere.SphereShape", "sphere", {}),
        ("_filemesh.FileMeshShape", "file_mesh", {}),
        ("_buffermesh.BufferMeshShape", "buffer_mesh", {}),
    ],
    cls_prefix="eradiate.scenes.shapes",
)


@parse_docs
@attrs.define(eq=False, slots=False)
class Shape:
    """
    Abstract interface for all shape scene elements.

    Notes
    -----
    * This class is to be used as a mixin.
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default="shape",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"shape"',
    )

    bsdf: t.Union[BSDFNode, Ref, None] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(bsdf_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((BSDF, Ref))
            ),
        ),
        doc="BSDF attached to the shape. If a dictionary is passed, it is "
        "interpreted by :class:`bsdf_factory.convert() <.Factory>`. "
        "If unset, no BSDF will be specified during the kernel dictionary "
        "generation: the kernel's default will be used.",
        type="BSDF or Ref or None",
        init_type="BSDF or Ref or dict, optional",
    )

    @property
    def objects(self) -> t.Optional[t.Dict[str, NodeSceneElement]]:
        if self.bsdf is None:
            return None
        else:
            return {"bsdf": self.bsdf}


@attrs.define(eq=False, slots=False)
class ShapeNode(Shape, NodeSceneElement, ABC):
    pass


@attrs.define(eq=False, slots=False)
class ShapeInstance(Shape, InstanceSceneElement, ABC):
    pass
