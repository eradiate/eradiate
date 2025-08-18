from __future__ import annotations

import attrs
import pint
import pinttrs

from ._core import AbstractDirectionalIllumination
from ..core import BoundingBox, NodeSceneElement, Ref
from ...attrs import define, documented
from ...units import unit_context_kernel as ucc
from ...units import unit_context_kernel as uck


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
                pinttrs.validators.has_compatible_units
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
                pinttrs.validators.has_compatible_units
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

    periodic_box: BoundingBox | Ref | None = documented(
        attrs.field(
            default=None,
            validator=attrs.validators.optional(
                attrs.validators.instance_of((BoundingBox, Ref))
            ),
            # converter=attrs.converters.optional(BoundingBox | Ref),
        ),
        doc="Bounding box defining the periodic boundary. Exclusive with "
        "`pbox_min` and `pbox_max`.",
        type=".BoundingBox or None",
        init_type=".BoundingBox or dict, optional",
        default="None",
    )

    @pbox_min.validator
    @pbox_max.validator
    def _pbox_validator(self, attribute, value):
        if self.pbox_min is not None and self.pbox_max is not None:
            if self.periodic_box is not None:
                raise ValueError(
                    "Cannot specify both 'pbox_min'/'pbox_max' and 'periodic_box' "
                    "at the same time."
                )

    @periodic_box.validator
    def _periodic_box_validator(self, attribute, value):
        if value is not None:
            if self.pbox_min is not None or self.pbox_max is not None:
                raise ValueError(
                    "Cannot specify both 'periodic_box' and 'pbox_min'/'pbox_max' "
                    "at the same time."
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

        elif self.periodic_box is not None:
            if isinstance(self.periodic_box, BoundingBox):
                result["pbox_min"] = self.periodic_box.min.m_as(uck.get("length"))
                result["pbox_max"] = self.periodic_box.max.m_as(uck.get("length"))
        else:
            raise ValueError(
                "Either 'pbox_min'/'pbox_max' or 'periodic_box' must be specified."
            )

        return result

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        result = {"irradiance": self.irradiance}
        if self.periodic_box is not None and isinstance(self.periodic_box, Ref):
            result["periodic_box"] = self.periodic_box

        return result
