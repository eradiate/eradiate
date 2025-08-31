from __future__ import annotations

import attrs

from ._core import AbstractDirectionalIllumination
from ..core import BoundingBox, NodeSceneElement
from ...attrs import define, documented
from ...units import unit_context_kernel as uck


@define(eq=False, slots=False)
class DirectionalPeriodicIllumination(AbstractDirectionalIllumination):
    """
    Directional periodic illumination scene element [``directionalperiodic``].

    This illumination source emits directional radiation from the top face of a
    periodic bounding box. The illumination direction is determined by zenith
    and azimuth angles following the Earth observation convention.

    Notes
    -----
    Currently only compatible with the :class:`.PAccumulatorIntergator`.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    periodic_box: BoundingBox = documented(
        attrs.field(
            factory=lambda: BoundingBox([-1, -1, -1], [1, 1, 1]),
            converter=BoundingBox.convert,
        ),
        doc="Bounding box of the periodic boundary. Rays are emitted from "
        "the top face of the this bounding box.",
        type=":class:`.BoundingBox`",
        init_type=":class:`.BoundingBox`, dict, tuple, or array-like, optional",
        default=None,
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def template(self) -> dict:
        result = {
            "type": "directionalperiodic",
            "to_world": self._to_world,
        }

        result["pbox_min"] = self.periodic_box.min.m_as(uck.get("length"))
        result["pbox_max"] = self.periodic_box.max.m_as(uck.get("length"))

        return result

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        result = {"irradiance": self.irradiance}
        return result
