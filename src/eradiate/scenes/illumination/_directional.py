from __future__ import annotations

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._core import Illumination
from ..core import NodeSceneElement
from ..spectra import SolarIrradianceSpectrum, Spectrum, spectrum_factory
from ..._config import config
from ...attrs import documented, parse_docs
from ...frame import AzimuthConvention, angles_to_direction
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import has_quantity, is_positive


@parse_docs
@attrs.define(eq=False, slots=False)
class DirectionalIllumination(Illumination):
    """
    Directional illumination scene element [``directional``].

    The illumination is oriented based on the classical angular convention used
    in Earth observation.
    """

    zenith: pint.Quantity = documented(
        pinttr.field(
            default=0.0 * ureg.deg,
            validator=[is_positive, pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Zenith angle.\n\nUnit-enabled field (default units: ucc[angle]).",
        type="quantity",
        init_type="quantity or float",
        default="0.0 deg",
    )

    azimuth: pint.Quantity = documented(
        pinttr.field(
            default=0.0 * ureg.deg,
            validator=[is_positive, pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle value.\n"
        "\n"
        "Unit-enabled field (default units: ucc[angle]).",
        type="quantity",
        init_type="quantity or float",
        default="0.0 deg",
    )

    azimuth_convention: AzimuthConvention = documented(
        attrs.field(
            default=None,
            converter=lambda x: config.azimuth_convention
            if x is None
            else (AzimuthConvention[x.upper()] if isinstance(x, str) else x),
            validator=attrs.validators.instance_of(AzimuthConvention),
        ),
        doc="Azimuth convention. If ``None``, the global default configuration "
        "is used (see :class:`.EradiateConfig`).",
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
        "Can be initialised with a dictionary processed by "
        ":meth:`.SpectrumFactory.convert`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        init_type=":class:`~eradiate.scenes.spectra.Spectrum` or dict or float",
        default=":class:`SolarIrradianceSpectrum() <.SolarIrradianceSpectrum>`",
    )

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

    @property
    def _to_world(self) -> mi.ScalarTransorm4f:
        direction = dr.normalize(mi.ScalarVector3f(self.direction))
        up, _ = mi.coordinate_system(direction)
        return mi.ScalarTransform4f.look_at(origin=0.0, target=direction, up=up)

    @property
    def template(self) -> dict:
        return {"type": "directional", "to_world": self._to_world}

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        return {"irradiance": self.irradiance}
