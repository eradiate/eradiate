from __future__ import annotations

import attrs

from ._core import Illumination
from ..core import NodeSceneElement
from ..spectra import Spectrum, spectrum_factory
from ...attrs import define, documented
from ...validators import has_quantity


@define(eq=False, slots=False)
class ConstantIllumination(Illumination):
    """
    Constant illumination scene element [``constant``].
    """

    radiance: Spectrum = documented(
        attrs.field(
            default=1.0,
            converter=spectrum_factory.converter("radiance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("radiance"),
            ],
        ),
        doc="Emitted radiance spectrum. Must be a radiance spectrum "
        "(in W/m²/sr/nm or compatible units).",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        init_type=":class:`~eradiate.scenes.spectra.Spectrum` or dict or float",
        default="1.0 ucc[radiance]",
    )

    @property
    def template(self) -> dict:
        return {"type": "constant"}

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        return {"radiance": self.radiance}
