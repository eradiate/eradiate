import attrs

from ._core import Illumination
from ..core import KernelDict
from ..spectra import Spectrum, spectrum_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...validators import has_quantity


@parse_docs
@attrs.define
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

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        return KernelDict(
            {
                self.id: {
                    "type": "constant",
                    "radiance": self.radiance.kernel_dict(ctx=ctx)["spectrum"],
                }
            }
        )
