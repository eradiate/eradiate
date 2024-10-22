from __future__ import annotations

import pint
import pinttrs

from ._core import BSDF
from ...attrs import define, documented
from ...kernel import InitParameter, UpdateParameter
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


@define(eq=False, slots=False)
class OceanLegacyBSDF(BSDF):
    """
    Ocean Legacy BSDF [``ocean_legacy``].

    This BSDF implements the 6SV ocean surface model as described in
    :cite:t:`Kotchenova:06`. This model treats the ocean as an opaque
    surface, and models the sunglint, whitecap and underlight components
    of the ocean reflectance. It depends on wind properties and
    pigmentation and chlorinity which makes it suitable to represent
    case I waters as defined by :cite:t:`Morel_1988`.

    See Also
    --------
    :ref:`plugin-bsdf-ocean_legacy`
    """

    wind_speed: pint.Quantity = documented(
        pinttrs.field(
            units=ureg("m/s").units,
            factory=lambda: 0.01 * ureg("m/s"),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Wind speed [m/s] at 10 meters above the surface.",
        type="quantity",
        init_type="quantity or float",
        default="0.01 m/s",
    )

    wind_direction: pint.Quantity = documented(
        pinttrs.field(
            factory=lambda: 0.0 * ureg.deg,
            validator=[is_positive, pinttrs.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Wind azimuthal angle.\n\nUnit-enabled field (default units: ucc['angle']).",
        type="quantity",
        init_type="quantity or float",
        default="0.0 deg",
    )

    chlorinity: pint.Quantity = documented(
        pinttrs.field(
            units=ureg("g/kg").units,
            factory=lambda: 19.0 * ureg("g/kg"),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Wind speed in m/s at 10 meters above the surface.",
        type="float or None",
        init_type="float, optional",
    )

    pigmentation: pint.Quantity = documented(
        pinttrs.field(
            units=ureg("mg/m^3").units,
            factory=lambda: 0.3 * ureg("mg/m^3"),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Wind speed in m/s at 10 meters above the surface.",
        type="float or None",
        init_type="float, optional",
    )

    def default_shininess(self, u: pint.Quantity):
        """
        Parametrizes the Blinn-Phong distribution function with respect to the
        wind speed.
        """
        return (37.2455 - u.m_as("m/s")) ** 1.15

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = {
            "type": "ocean_legacy",
            "wavelength": InitParameter(lambda ctx: ctx.si.w.m_as("nm")),
            "shininess": self.default_shininess(self.wind_speed),
            "wind_speed": self.wind_speed.m_as("m/s"),
            "wind_direction": self.wind_direction.m_as("rad"),
            "chlorinity": self.chlorinity.m_as("g/kg"),
            "pigmentation": self.pigmentation.m_as("mg/m^3"),
        }

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, UpdateParameter]:
        result = {"wavelength": UpdateParameter(lambda ctx: ctx.si.w.m_as("nm"))}

        return result
