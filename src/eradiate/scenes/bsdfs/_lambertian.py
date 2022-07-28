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
class LambertianBSDF(BSDF):
    """
    Lambertian BSDF [``lambertian``].

    This class implements the Lambertian (a.k.a. diffuse) reflectance model.
    A surface with this scattering model attached scatters radiation equally in
    every direction.

    Notes
    -----
    This is a thin wrapper around the ``diffuse`` kernel plugin.
    """

    reflectance: Spectrum = documented(
        attr.ib(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.5",
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        return KernelDict(
            {
                self.id: {
                    "type": "diffuse",
                    "reflectance": onedict_value(self.reflectance.kernel_dict(ctx=ctx)),
                }
            }
        )
