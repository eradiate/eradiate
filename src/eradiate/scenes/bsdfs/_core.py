from __future__ import annotations

import typing as t
from abc import ABC

import attrs

from ..core import NodeSceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc

bsdf_factory = Factory()
bsdf_factory.register_lazy_batch(
    [
        ("_black.BlackBSDF", "black", {}),
        ("_checkerboard.CheckerboardBSDF", "checkerboard", {}),
        ("_lambertian.LambertianBSDF", "lambertian", {}),
        ("_mqdiffuse.MQDiffuseBSDF", "mqdiffuse", {}),
        ("_rpv.RPVBSDF", "rpv", {}),
    ],
    cls_prefix="eradiate.scenes.bsdfs",
)


@attrs.define(eq=False, slots=False)
class BSDF:
    """
    Interface for all BSDF scene elements.

    Notes
    -----
    * This class is to be used as a mixin.
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default="bsdf",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"bsdf"',
    )
