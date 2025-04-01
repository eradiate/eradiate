from __future__ import annotations

import attrs
import pint
import pinttrs

from ._core import BSDF
from ...attrs import define, documented
from ...kernel import DictParameter, SceneParameter
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


@define(eq=False, slots=False)
class OceanLegacyBSDF(BSDF):
    """
    Ocean Legacy BSDF [``ocean_legacy``].

    This BSDF implements the 6SV ocean surface model as described in
    :cite:t:`Kotchenova2006The6SVRTM`. This model treats the ocean as an opaque
    surface, and models the sunglint, whitecap and underlight components
    of the ocean reflectance. It depends on wind properties and
    pigmentation and chlorinity which makes it suitable to represent
    case I waters as defined by :cite:t:`Morel1988ModelingUpperOcean`.

    See Also
    --------
    :ref:`plugin-bsdf-ocean_legacy`

    Notes
    -----
     The ``wind_direction`` parameter indicates the azimuth angle of the wind and
    is interpreted using the :ref:`North left convention <sec-user_guide-conventions-azimuth>`.
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
        doc="Wind azimuthal angle in *North Left-Hand convention*.\n\nUnit-enabled field (default units: ucc['angle']).",
        type="quantity",
        init_type="quantity or float",
        default="0.0 deg",
    )

    chlorinity: pint.Quantity = documented(
        pinttrs.field(
            units=ureg.Unit("g/kg"),
            factory=lambda: 19.0 * ureg("g/kg"),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Chlorinity of water.",
        type="quantity",
        init_type="quantity or float",
        default="19.0 g/kg",
    )

    pigmentation: pint.Quantity = documented(
        pinttrs.field(
            units=ureg.Unit("mg/m^3"),
            factory=lambda: 0.3 * ureg("mg/m^3"),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Pigmentation of water.",
        type="quantity",
        init_type="quantity or float",
        default="0.3 mg/m^3",
    )

    shadowing: bool = documented(
        attrs.field(
            converter=bool,
            validator=attrs.validators.instance_of(bool),
            default=True,
        ),
        doc="Indicates whether evaluation of BRDF computes shadowing and masking.",
        type="bool",
        default="True",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = {
            "type": "ocean_legacy",
            "wavelength": DictParameter(lambda ctx: ctx.si.w.m_as("nm")),
            "wind_speed": self.wind_speed.m_as("m/s"),
            "wind_direction": self.wind_direction.m_as("deg"),
            "chlorinity": self.chlorinity.m_as("g/kg"),
            "pigmentation": self.pigmentation.m_as("mg/m^3"),
            "shadowing": self.shadowing,
        }

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        result = {"wavelength": SceneParameter(lambda ctx: ctx.si.w.m_as("nm"))}

        return result
