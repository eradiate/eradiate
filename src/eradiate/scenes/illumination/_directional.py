from __future__ import annotations

import attrs
import mitsuba as mi

from ._core import AbstractDirectionalIllumination
from ... import traverse
from ...attrs import define
from ...kernel import SearchSceneParameter


@define(eq=False, slots=False)
class DirectionalIllumination(AbstractDirectionalIllumination):
    """
    Directional illumination scene element [``directional``].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.
    """

    @property
    def template(self) -> dict:
        result = {"type": "directional", "to_world": self._to_world, "id": self.id}
        kdict, _ = traverse(self.irradiance)
        for k, v in kdict.items():
            result[f"irradiance.{k}"] = v
        return result

    @property
    def params(self):
        _, kpmap_irradiance = traverse(self.irradiance)

        result = {}
        for key, param in kpmap_irradiance.items():
            result[f"irradiance.{key}"] = attrs.evolve(
                param,
                tracks=SearchSceneParameter(
                    node_type=mi.Emitter,
                    node_id=self.id,
                    parameter_relpath=f"irradiance.{param.tracks.strip()}",
                )
                if isinstance(param.tracks, str)
                else param,
            )

        return result
