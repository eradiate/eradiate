from __future__ import annotations

import warnings
from abc import ABC

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import pint
import pinttrs

from ..core import NodeSceneElement
from ..spectra import SolarIrradianceSpectrum, Spectrum, spectrum_factory
from ..._factory import Factory
from ...attrs import define, documented, get_doc
from ...config import settings
from ...frame import AzimuthConvention, angles_to_direction
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import has_quantity, is_positive

illumination_factory = Factory()
illumination_factory.register_lazy_batch(
    [
        ("_astro_object.AstroObjectIllumination", "astro_object", {}),
        ("_constant.ConstantIllumination", "constant", {}),
        ("_directional.DirectionalIllumination", "directional", {}),
        ("_spot.SpotIllumination", "spot", {}),
    ],
    cls_prefix="eradiate.scenes.illumination",
)


@define(eq=False, slots=False)
class Illumination(NodeSceneElement, ABC):
    """
    Abstract base class for all illumination scene elements.

    Notes
    -----
    * This class is to be used as a mixin.
    """

    id: str | None = documented(
        attrs.field(
            default="illumination",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(NodeSceneElement, "id", "doc"),
        type=get_doc(NodeSceneElement, "id", "type"),
        init_type=get_doc(NodeSceneElement, "id", "init_type"),
        default='"illumination"',
    )


def _azimuth_converter(value):
    if not 0.0 <= value.m_as("deg") < 360.0:
        warnings.warn(
            "Illumination azimuth values should be in the [0°, 360°[ interval. "
            "Applying modulo operation."
        )
        return value % (360.0 * ureg.deg)
    else:
        return value


@define(eq=False, slots=False)
class AbstractDirectionalIllumination(Illumination, ABC):
    """
    Abstract interface to directional-like illuminants.
    """

    zenith: pint.Quantity = documented(
        pinttrs.field(
            default=0.0 * ureg.deg,
            validator=[is_positive, pinttrs.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Zenith angle.\n\nUnit-enabled field (default units: ucc['angle']).",
        type="quantity",
        init_type="quantity or float",
        default="0.0 deg",
    )

    azimuth: pint.Quantity = documented(
        pinttrs.field(
            default=0.0 * ureg.deg,
            converter=[
                pinttrs.converters.to_units(ucc.deferred("angle")),
                _azimuth_converter,
            ],
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle value.\n\nUnit-enabled field (default units: ucc['angle']).",
        type="quantity",
        init_type="quantity or float",
        default="0.0 deg",
    )

    azimuth_convention: AzimuthConvention = documented(
        attrs.field(
            default=None,
            converter=lambda x: (
                settings.azimuth_convention
                if x is None
                else (AzimuthConvention[x.upper()] if isinstance(x, str) else x)
            ),
            validator=attrs.validators.instance_of(AzimuthConvention),
        ),
        doc="Azimuth convention. If ``None``, the global default configuration "
        "is used (see :ref:`sec-user_guide-config`).",
        type=".AzimuthConvention",
        init_type=".AzimuthConvention or str, optional",
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

    @property
    def _to_world(self) -> mi.ScalarTransorm4f:
        direction = dr.normalize(mi.ScalarVector3f(self.direction))
        up, _ = mi.coordinate_system(direction)
        return mi.ScalarTransform4f.look_at(origin=0.0, target=direction, up=up)

    @property
    def direction(self) -> np.ndarray:
        """
        Illumination direction as an array of shape (3,), pointing inwards.
        """
        return angles_to_direction(
            [self.zenith.m_as(ureg.rad), self.azimuth.m_as(ureg.rad)],
            azimuth_convention=self.azimuth_convention,
            flip=True,
        ).reshape((3,))
