import attrs

from ._core import BSDFNode


@attrs.define(eq=False, slots=False)
class BlackBSDF(BSDFNode):
    """
    Black BSDF [``black``].

    This BSDF models a perfectly absorbing surface. It is equivalent to a
    :class:`.DiffuseBSDF` with zero reflectance.
    """

    @property
    def template(self) -> dict:
        return {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.0},
        }
