from __future__ import annotations

from abc import ABC

import attrs

from ..core import NodeSceneElement
from ..._factory import Factory
from ...attrs import parse_docs

bsdf_factory = Factory()
bsdf_factory.register_lazy_batch(
    [
        ("_black.BlackBSDF", "black", {}),
        ("_checkerboard.CheckerboardBSDF", "checkerboard", {}),
        ("_lambertian.LambertianBSDF", "lambertian", {}),
        ("_mqdiffuse.MQDiffuseBSDF", "mqdiffuse", {}),
        ("_opacity_mask.OpacityMaskBSDF", "opacity_mask", {}),
        ("_rpv.RPVBSDF", "rpv", {}),
    ],
    cls_prefix="eradiate.scenes.bsdfs",
)


@parse_docs
@attrs.define(eq=False, slots=False)
class BSDF(NodeSceneElement, ABC):
    """
    Abstract base class  for all BSDF scene elements.
    """

    pass
