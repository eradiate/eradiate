import attr
import mitsuba as mi

from ._core import BSDF, bsdf_factory
from ..core import KernelDict
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext


@bsdf_factory.register(type_id="checkerboard")
@parse_docs
@attr.s
class CheckerboardBSDF(BSDF):
    """
    Checkerboard BSDF [``checkerboard``].

    This class defines a Lambertian BSDF textured with a checkerboard pattern.
    """

    reflectance_a: Spectrum = documented(
        attr.ib(
            default=0.2,
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
        default="0.2",
    )

    reflectance_b: Spectrum = documented(
        attr.ib(
            default=0.8,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=":class:`.Spectrum` or dict or float",
        default="0.8",
    )

    scale_pattern: float = documented(
        attr.ib(
            default=2.0, converter=float, validator=attr.validators.instance_of(float)
        ),
        doc="Scaling factor for the checkerboard pattern. The higher the value, "
        "the more checkboard patterns will fit on the surface to which this "
        "reflection model is attached.",
        type="float",
        default=2.0,
    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        return KernelDict(
            {
                self.id: {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "checkerboard",
                        "color0": self.reflectance_a.kernel_dict(ctx=ctx)["spectrum"],
                        "color1": self.reflectance_b.kernel_dict(ctx=ctx)["spectrum"],
                        "to_uv": mi.ScalarTransform4f.scale(self.scale_pattern),
                    },
                }
            }
        )
