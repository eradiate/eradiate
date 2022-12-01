import typing as t

import attrs

from ._core import Illumination
from ..core import NodeSceneElement
from ..spectra import Spectrum, spectrum_factory
from ...attrs import documented, parse_docs
from ...validators import has_quantity


@parse_docs
@attrs.define(eq=False)
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
    def kernel_type(self) -> str:
        return "constant"

    @property
    def objects(self) -> t.Dict[str, NodeSceneElement]:
        return {"radiance": self.radiance}
