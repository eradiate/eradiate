from typing import Dict

import attr

from ._core import Surface, surface_factory
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ..._util import onedict_value
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext


@surface_factory.register(type_id="rpv")
@parse_docs
@attr.s
class RPVSurface(Surface):
    """
    RPV surface scene element [``rpv``].

    This class creates a square surface to which a RPV BRDF
    :cite:`Rahman1993CoupledSurfaceatmosphereReflectance,Pinty2000SurfaceAlbedoRetrieval`
    is attached.

    The default configuration corresponds to grassland (visible light)
    (:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`, Table 1).

    .. note:: Parameter names are defined as per the symbols used in the
       Eradiate Scientific Handbook :cite:`EradiateScientificHandbook2020`.
    """

    rho_0: Spectrum = documented(
        attr.ib(
            default=0.183,
            converter=spectrum_factory.converter("dimensionless"),
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

    rho_c: Spectrum = documented(
        attr.ib(
            default=0.183,
            converter=spectrum_factory.converter("dimensionless"),
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

    k: Spectrum = documented(
        attr.ib(
            default=0.780,
            converter=spectrum_factory.converter("dimensionless"),
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

    g: Spectrum = documented(
        attr.ib(
            default=-0.1,
            converter=spectrum_factory.converter("dimensionless"),
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

    def bsdfs(self, ctx: KernelDictContext) -> Dict:
        return {
            f"bsdf_{self.id}": {
                "type": "rpv",
                "rho_0": onedict_value(self.rho_0.kernel_dict(ctx)),
                "rho_c": onedict_value(self.rho_c.kernel_dict(ctx)),
                "k": onedict_value(self.k.kernel_dict(ctx)),
                "g": onedict_value(self.g.kernel_dict(ctx)),
            }
        }
