from __future__ import annotations

import typing as t
from abc import ABC

import attrs

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc

bsdf_factory = Factory()
bsdf_factory.register_lazy_batch(
    [
        ("_black.BlackBSDF", "black", {}),
        ("_checkerboard.CheckerboardBSDF", "checkerboard", {}),
        ("_lambertian.LambertianBSDF", "lambertian", {}),
        ("_rpv.RPVBSDF", "rpv", {}),
    ],
    cls_prefix="eradiate.scenes.bsdfs",
)


@attrs.define
class BSDF(SceneElement, ABC):
    """
    Abstract interface for all BSDF scene elements.
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default="bsdf",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"bsdf"',
    )
