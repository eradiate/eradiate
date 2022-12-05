import attrs

from ._core import BSDF
from ..core import NodeSceneElement


@attrs.define(eq=False, slots=False)
class BlackBSDF(BSDF, NodeSceneElement):
    """
    Black BSDF [``black``].

    This BSDF models a perfectly absorbing surface. It is equivalent to a
    :class:`.DiffuseBSDF` with zero reflectance.
    """

    @property
    def kernel_type(self) -> str:
        return "diffuse"

    @property
    def template(self) -> dict:
        return {**super().template, "reflectance": {"type": "uniform", "value": 0.0}}
