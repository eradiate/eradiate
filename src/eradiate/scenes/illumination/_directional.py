from __future__ import annotations

from ._core import AbstractDirectionalIllumination
from ... import traverse
from ...attrs import define
from ...kernel import SearchSceneParameter, scene_parameter


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
    def params(self):
        import mitsuba as mi

        _, kpmap_irradiance = traverse(self.irradiance)

        result = {}
        for key, param in kpmap_irradiance.items():
            if isinstance(param.tracks, str):
                tracks = SearchSceneParameter(
                    node_type=mi.Emitter,
                    node_id=self.id,
                    parameter_relpath=f"irradiance.{param.tracks.strip()}",
                )
            else:
                # Safety guard, should not happen as all Spectrum implementations
                # currently assume anonymous nodes
                raise NotImplementedError()

            result[f"irradiance.{key}"] = scene_parameter(
                flags=param.flags, tracks=tracks
            )(param.func)

        return result
