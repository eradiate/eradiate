from __future__ import annotations

import attrs

from ._core import BSDF
from ...kernel import SceneParameter


@attrs.define(eq=False, slots=False)
class BlackBSDF(BSDF):
    """
    Black BSDF [``black``].

    This BSDF models a perfectly absorbing surface. It is equivalent to a
    :class:`.DiffuseBSDF` with zero reflectance.
    """

    @property
    def template(self) -> dict:
        # Inherit docstring

        result = {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.0},
        }

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        return {}
