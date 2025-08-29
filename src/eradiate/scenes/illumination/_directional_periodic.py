from __future__ import annotations

import attrs
import pint
import pinttrs

from ._core import AbstractDirectionalIllumination
from ..core import NodeSceneElement
from ...attrs import define, documented
from ...units import unit_context_kernel as ucc
from ...units import unit_context_kernel as uck
from ...validators import is_vector3, on_quantity


@define(eq=False, slots=False)
class DirectionalPeriodicIllumination(AbstractDirectionalIllumination):
    """
    Directional periodic illumination scene element [``directionalperiodic``].

    This illumination source emits directional radiation from the top face of a
    periodic bounding box. The illumination direction is determined by zenith
    and azimuth angles following the Earth observation convention.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    pbox_min: pint.Quantity | None = documented(
        pinttrs.field(
            default=None,
            validator=attrs.validators.optional(
                (pinttrs.validators.has_compatible_units, on_quantity(is_vector3))
            ),
            units=ucc.deferred("length"),
        ),
        doc="Minimum point of the periodic bounding box. Must be used together with "
        "`pbox_max`. Exclusive with `periodic_box`.\n\n"
        "Unit-enabled field (default units: ucc['length']).",
        type="quantity or None",
        init_type="quantity or array-like, optional",
        default="None",
    )

    pbox_max: pint.Quantity | None = documented(
        pinttrs.field(
            default=None,
            validator=attrs.validators.optional(
                (pinttrs.validators.has_compatible_units, on_quantity(is_vector3))
            ),
            units=ucc.deferred("length"),
        ),
        doc="Maximum point of the periodic bounding box. Must be used together with "
        "`pbox_min`. Exclusive with `periodic_box`.\n\n"
        "Unit-enabled field (default units: ucc['length']).",
        type="quantity or None",
        init_type="quantity or array-like, optional",
        default="None",
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

        if self.pbox_min is not None and self.pbox_max is not None:
            result["pbox_min"] = self.pbox_min.m_as(uck.get("length"))
            result["pbox_max"] = self.pbox_max.m_as(uck.get("length"))
        else:
            raise ValueError(
                "Either 'pbox_min'/'pbox_max' or 'periodic_box' must be specified."
            )

        return result

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        result = {"irradiance": self.irradiance}
        return result
