from __future__ import annotations

import attrs
import pint
import pinttr

from ._directional import DirectionalIllumination
from ...attrs import documented, parse_docs
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


@parse_docs
@attrs.define(eq=False, slots=False)
class AstroObjectIllumination(DirectionalIllumination):

    angular_diameter: pint.Quantity = documented(
        pinttr.field(
            default=1.0 * ureg.deg,
            validator=[is_positive, pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Angular diameter of the AstroObject, as scene from the observation point."
            "\n\nUnit-enabled field (default units: ucc[angle]).",
        type="quantity",
        init_type="quantity or float",
        default="1 deg",
    )

    @classmethod
    def astro_object(cls, **kwargs):

        return cls(**kwargs)

    @property
    def template(self) -> dict:
        return {"type": "astroobject", "to_world": self._to_world, "angular_diameter": self.angular_diameter.m_as("degree")}
