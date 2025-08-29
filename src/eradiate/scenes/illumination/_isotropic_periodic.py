from __future__ import annotations

import attrs

from ._core import Illumination
from ..core import BoundingBox, NodeSceneElement
from ..spectra import SolarIrradianceSpectrum, Spectrum, spectrum_factory
from ...attrs import define, documented
from ...units import unit_context_kernel as uck
from ...validators import has_quantity


@define(eq=False, slots=False)
class IsotropicPeriodicIllumination(Illumination):
    """
    Isotropic periodic illumination scene element [``directionalperiodic``].

    This illumination source emits radiation from the top face of a periodic
    bounding box in the downwelling hemisphere isotropically.

    Note:
    -----
    Currently only compatible with the :class:`.PAccumulatorIntergator`.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    periodic_box: BoundingBox = documented(
        attrs.field(
            factory=lambda: BoundingBox([-1, -1, -1], [1, 1, 1]),
            converter=BoundingBox.convert,
        ),
        doc="Bounding box of the periodic boundary. Rays are emitted from "
        "the top face of the this bounding box.",
        type=":class:`.BoundingBox`",
        init_type=":class:`.BoundingBox`, dict, tuple, or array-like, optional",
        default=None,
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

        result["pbox_min"] = self.periodic_box.min.m_as(uck.get("length"))
        result["pbox_max"] = self.periodic_box.max.m_as(uck.get("length"))

        return result

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        result = {"irradiance": self.irradiance}
        return result
