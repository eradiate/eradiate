import typing as t

import attr

from ._core import BSDF
from ..core import KernelDict
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...util.misc import onedict_value


@parse_docs
@attr.s
class RPVBSDF(BSDF):
    """
    RPV BSDF [``rpv``].

    This BSDF implements the Rahman-Pinty-Verstraete (RPV) reflection model
    :cite:`Rahman1993CoupledSurfaceatmosphereReflectance,Pinty2000SurfaceAlbedoRetrieval`.
    It notably features a controllable back-scattering lobe (`hot spot`)
    characteristic of many natural land surfaces and is frequently used in Earth
    observation because of its simple parametrisation.

    See Also
    --------
    :ref:`plugin-bsdf-rpv`

    Notes
    -----
    * The default configuration is typical of grassland in the visible domain
      (:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`, Table 1).
    * Parameter names are defined as per the symbols used in the Eradiate
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
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="0.183",
    )

    rho_c: t.Optional[Spectrum] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(
                spectrum_factory.converter("dimensionless")
            ),
            validator=attr.validators.optional(
                [
                    attr.validators.instance_of(Spectrum),
                    validators.has_quantity("dimensionless"),
                ]
            ),
        ),
        doc="Hot spot parameter. Must be dimensionless. "
        r"Should be in :math:`[0, 1]`. If unset, :math:`\rho_\mathrm{c}` "
        r"defaults to the kernel plugin default (equal to :math:`\rho_0`).",
        type=".Spectrum or None",
        init_type=".Spectrum or dict or float or None, optional",
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
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
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
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="-0.1",
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        result = {
            self.id: {
                "type": "rpv",
                "rho_0": onedict_value(self.rho_0.kernel_dict(ctx)),
                "k": onedict_value(self.k.kernel_dict(ctx)),
                "g": onedict_value(self.g.kernel_dict(ctx)),
            }
        }

        if self.rho_c is not None:
            result[self.id]["rho_c"] = onedict_value(self.rho_c.kernel_dict(ctx))

        return KernelDict(result)
