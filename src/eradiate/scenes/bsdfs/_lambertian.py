import typing as t

import attrs

from ._core import BSDF
from ..core import NodeSceneElement
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import documented, parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class LambertianBSDF(NodeSceneElement, BSDF):
    """
    Lambertian BSDF [``lambertian``].

    This class implements the Lambertian (a.k.a. diffuse) reflectance model.
    A surface with this scattering model attached scatters radiation equally in
    every direction.
    """

    reflectance: Spectrum = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.5",
    )

    @property
    def kernel_type(self) -> str:
        return "diffuse"

    @property
    def objects(self) -> t.Dict[str, NodeSceneElement]:
        return {"reflectance": self.reflectance}
