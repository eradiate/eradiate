from typing import MutableMapping, Optional

import attr

from ._core import Illumination, IlluminationFactory
from ..spectra import Spectrum, SpectrumFactory
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...validators import has_quantity


@IlluminationFactory.register("constant")
@parse_docs
@attr.s
class ConstantIllumination(Illumination):
    """
    Constant illumination scene element [:factorykey:`constant`].
    """

    radiance = documented(
        attr.ib(
            default=1.0,
            converter=SpectrumFactory.converter("radiance"),
            validator=[attr.validators.instance_of(Spectrum), has_quantity("radiance")],
        ),
        doc="Emitted radiance spectrum. Must be a radiance spectrum "
        "(in W/m^2/sr/nm or compatible units).",
        type="float or :class:`~eradiate.scenes.spectra.Spectrum`",
        default="1.0 ucc[radiance]",
    )

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        return {
            self.id: {
                "type": "constant",
                "radiance": self.radiance.kernel_dict(ctx=ctx)["spectrum"],
            }
        }
