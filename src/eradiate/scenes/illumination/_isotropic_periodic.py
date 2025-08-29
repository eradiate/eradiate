from __future__ import annotations

import attrs
import pint
import pinttrs

from ._core import Illumination
from ..core import NodeSceneElement
from ..spectra import SolarIrradianceSpectrum, Spectrum, spectrum_factory
from ...attrs import define, documented
from ...units import unit_context_kernel as ucc
from ...units import unit_context_kernel as uck
from ...validators import has_quantity, is_vector3, on_quantity


@define(eq=False, slots=False)
class IsotropicPeriodicIllumination(Illumination):
    """
    Isotropic periodic illumination scene element [``directionalperiodic``].

    This illumination source emits isotropic radiation from the top face of a
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
        "`pbox_max`.\n\n"
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
        "`pbox_min`.\n\n"
        "Unit-enabled field (default units: ucc['length']).",
        type="quantity or None",
        init_type="quantity or array-like, optional",
        default="None",
    )

    irradiance: Spectrum = documented(
        attrs.field(
            factory=SolarIrradianceSpectrum,
            converter=spectrum_factory.converter("irradiance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("irradiance"),
            ],
        ),
        doc="Emitted power flux in the plane orthogonal to the illumination direction. "
        "Must be an irradiance spectrum (in W/m²/nm or compatible unit). "
        "Can be initialized with a dictionary processed by "
        ":meth:`.SpectrumFactory.convert`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        init_type=":class:`~eradiate.scenes.spectra.Spectrum` or dict or float",
        default=":class:`SolarIrradianceSpectrum() <.SolarIrradianceSpectrum>`",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def template(self) -> dict:
        result = {
            "type": "isotropicperiodic",
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
