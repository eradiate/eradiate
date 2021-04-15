import attr

from ._core import Surface, SurfaceFactory
from ..spectra import Spectrum, SpectrumFactory
from ... import validators
from ..._attrs import documented, parse_docs


@SurfaceFactory.register("lambertian")
@parse_docs
@attr.s
class LambertianSurface(Surface):
    """Lambertian surface scene element [:factorykey:`lambertian`].

    This class creates a square surface to which a Lambertian BRDF is attached.
    """

    reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :class:`.SpectrumFactory`.",
        type=":class:`.UniformSpectrum`",
        default="0.5",
    )

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": self.reflectance.kernel_dict()["spectrum"],
            }
        }
