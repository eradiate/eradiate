import attr

from ._core import Surface, SurfaceFactory
from ..spectra import Spectrum, SpectrumFactory
from ... import validators
from ..._util import onedict_value
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext


@SurfaceFactory.register("rpv")
@parse_docs
@attr.s
class RPVSurface(Surface):
    """
    RPV surface scene element [:factorykey:`rpv`].

    This class creates a square surface to which a RPV BRDF
    :cite:`Rahman1993CoupledSurfaceatmosphereReflectance,Pinty2000SurfaceAlbedoRetrieval`
    is attached.

    The default configuration corresponds to grassland (visible light)
    (:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`, Table 1).

    .. note:: Parameter names are defined as per the symbols used in the
       Eradiate Scientific Handbook :cite:`EradiateScientificHandbook2020`.
    """

    rho_0 = documented(
        attr.ib(
            default=0.183,
            converter=SpectrumFactory.converter("dimensionless"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Amplitude parameter. Must be dimensionless. "
        "Should be in :math:`[0, 1]`.",
        type=":class:`.Spectrum`",
        default="0.183",
    )

    rho_c = documented(
        attr.ib(
            default=0.183,
            converter=SpectrumFactory.converter("dimensionless"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Hot spot parameter. Must be dimensionless. "
        "Should be in :math:`[0, 1]`.",
        type=":class:`.Spectrum`",
        default="0.183",
    )

    k = documented(
        attr.ib(
            default=0.780,
            converter=SpectrumFactory.converter("dimensionless"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Bowl-shape parameter. Must be dimensionless. "
        "Should be in :math:`[0, 2]`.",
        type=":class:`.Spectrum`",
        default="0.780",
    )

    g = documented(
        attr.ib(
            default=-0.1,
            converter=SpectrumFactory.converter("dimensionless"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Asymmetry parameter. Must be dimensionless. "
        "Should be in :math:`[-1, 1]`.",
        type=":class:`.Spectrum`",
        default="-0.1",
    )

    def bsdfs(self, ctx: KernelDictContext = None):
        return {
            f"bsdf_{self.id}": {
                "type": "rpv",
                "rho_0": onedict_value(self.rho_0.kernel_dict(ctx=ctx)),
                "rho_c": onedict_value(self.rho_c.kernel_dict(ctx=ctx)),
                "k": onedict_value(self.k.kernel_dict(ctx=ctx)),
                "g": onedict_value(self.g.kernel_dict(ctx=ctx)),
            }
        }
