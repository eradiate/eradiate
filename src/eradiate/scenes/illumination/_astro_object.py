from __future__ import annotations

import numpy as np
import pint
import pinttr

from ._core import AbstractDirectionalIllumination
from ..core import NodeSceneElement
from ...attrs import define, documented
from ...frame import angles_to_direction
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


@define(eq=False, slots=False)
class AstroObjectIllumination(AbstractDirectionalIllumination):
    """
    Astronomical Object Illumination scene element [``astro_object``].

    This illumination represents the light coming from a distant astronomical
    object (*e.g.* the Sun). The astronomical object uniformly illuminates a
    portion of the sky such that it appears to the observer as a circle. Its
    size is defined by the apparent diameter of the circle, expressed in
    degrees.

    Warnings
    --------
    This is an experimental feature. At the moment, using the
    :class:`.DirectionalIllumination` is recommended.

    Notes
    -----
    Contrary to the directional illuminant, the astronomical object is not a
    delta emitter, *i.e.* it has a non-zero apparent size. Light rays cast by
    this illuminant are, from the point of view of the observer, encompassed
    in a cone, while they are parallel in the case of the directional
    illumination.
    """

    angular_diameter: pint.Quantity = documented(
        pinttr.field(
            default=0.5358 * ureg.deg,
            validator=[is_positive, pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Apparent diameter of the celestial body as seen from the point. "
        "The default value is an average of the apparent diameter of the "
        "Sun as seen from Earth.\n"
        "\n"
        "Unit-enabled field (default units: ucc['angle']).",
        type="quantity",
        init_type="quantity or float",
        default="0.5358 deg",
    )

    @property
    def direction(self) -> np.ndarray:
        """
        Illumination direction as an array of shape (3,), pointing inwards.
        """
        return angles_to_direction(
            [self.zenith.m_as(ureg.rad), self.azimuth.m_as(ureg.rad)],
            azimuth_convention=self.azimuth_convention,
            flip=False,
        ).reshape((3,))

    @property
    def template(self) -> dict:
        # Inherit docstring
        return {
            "type": "astroobject",
            "to_world": self._to_world,
            "angular_diameter": self.angular_diameter.m_as("degree"),
        }

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        return {"irradiance": self.irradiance}
