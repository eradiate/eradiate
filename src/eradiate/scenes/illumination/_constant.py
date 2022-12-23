import typing as t

import attrs

from ._core import IlluminationNode
from ..core import NodeSceneElement
from ..spectra import Spectrum, SpectrumNode, spectrum_factory
from ...attrs import documented, parse_docs
from ...validators import has_quantity


@parse_docs
@attrs.define(eq=False, slots=False)
class ConstantIllumination(IlluminationNode):
    """
    Constant illumination scene element [``constant``].
    """

    radiance: SpectrumNode = documented(
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
    def objects(self) -> t.Dict[str, NodeSceneElement]:
        return {"radiance": self.radiance}
