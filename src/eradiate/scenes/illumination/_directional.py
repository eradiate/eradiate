from __future__ import annotations

from ._core import AbstractDirectionalIllumination
from ..core import NodeSceneElement
from ...attrs import define


@define(eq=False, slots=False)
class DirectionalIllumination(AbstractDirectionalIllumination):
    """
    Directional illumination scene element [``directional``].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.
    """

    @property
    def template(self) -> dict:
        return {"type": "directional", "to_world": self._to_world}

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        return {"irradiance": self.irradiance}
