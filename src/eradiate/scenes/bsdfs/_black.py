import attrs

from ._core import BSDF


@attrs.define(eq=False)
class BlackBSDF(BSDF):
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
