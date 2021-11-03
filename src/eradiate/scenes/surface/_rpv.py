import typing as t

import attr

from ._core import Surface, surface_factory
from ..core import KernelDict
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

    Notes
    -----
    Parameter names are defined as per the symbols used in the Eradiate
    Scientific Handbook :cite:`EradiateScientificHandbook2020`.
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
        init_type=":class:`.Spectrum` or dict or float, optional",
        default="0.183",
    )

    rho_c: t.Optional[Spectrum] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(
                spectrum_factory.converter("dimensionless")
            ),
            validator=attr.validators.optional(
                (
                    attr.validators.instance_of(Spectrum),
                    validators.has_quantity("dimensionless"),
                )
            ),
        ),
        doc="Hot spot parameter. Must be dimensionless. "
        r"Should be in :math:`[0, 1]`. If unset, :math:`\rho_\mathrm{c}` "
        r"defaults to the kernel plugin default (equal to :math:`\rho_0`).",
        type=":class:`.Spectrum` or None",
        init_type=":class:`.Spectrum` or dict or float or None, optional",
        default="None",
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
        init_type=":class:`.Spectrum` or dict or float, optional",
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
        init_type=":class:`.Spectrum` or dict or float, optional",
        default="-0.1",
    )

    def bsdfs(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict(
            {
                f"bsdf_{self.id}": {
                    "type": "rpv",
                    "rho_0": onedict_value(self.rho_0.kernel_dict(ctx)),
                    "k": onedict_value(self.k.kernel_dict(ctx)),
                    "g": onedict_value(self.g.kernel_dict(ctx)),
                }
            }
        )

        if self.rho_c is not None:
            result.data["rho_c"] = onedict_value(self.rho_c.kernel_dict(ctx))

        return result
