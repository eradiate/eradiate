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

    """
    Astronomical Object Illumination scene element [``astroobject``].

    This illumination represents the light coming from an astronomical object
    (e.g. the Sun). Contrary to the ``directional`` illumination, the
    astronomical object has a finite size and is not a point source. The
    illumination emulates the behavior of a planet/star, being a extremely far
    away illumination source, but still visible from the observation point.

    The illumination is oriented based on the classical angular convention used
    in Earth observation. It features a default angular diameter of 1 degree.
    The angular diameter controls the size of the astronomical object as seen
    from the observation point.

    Rays casted by the astronomical object are not parallel, but belong to a
    cone. The cone is defined by the angular diameter of the astronomical object
    and the angle between the interaction point and the center of the cone cap.
    """

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
