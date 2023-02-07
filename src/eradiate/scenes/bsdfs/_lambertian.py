import typing as t

import attrs
import mitsuba as mi
from rich.pretty import pprint

from ._core import BSDF
from ..core import traverse
from ..spectra import SpectrumNode, spectrum_factory
from ... import validators
from ...attrs import documented, parse_docs
from ...kernel import TypeIdLookupStrategy, UpdateParameter


@parse_docs
@attrs.define(eq=False, slots=False)
class LambertianBSDF(BSDF):
    """
    Lambertian BSDF [``lambertian``].

    This class implements the Lambertian (a.k.a. diffuse) reflectance model.
    A surface with this scattering model attached scatters radiation equally in
    every direction.
    """

    reflectance: SpectrumNode = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(SpectrumNode),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".SpectrumNode",
        init_type=".SpectrumNode or dict or float",
        default="0.5",
    )

    @property
    def template(self) -> dict:
        return {
            "type": "diffuse",
            **{
                f"reflectance.{key}": value
                for key, value in traverse(self.reflectance)[0].items()
            },
        }

    @property
    def params(self) -> t.Dict[str, UpdateParameter]:
        params = traverse(self.reflectance)[1].data

        result = {}
        for key, param in params.items():
            result[f"reflectance.{key}"] = attrs.evolve(
                param,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF,
                    node_id=self.id,
                    parameter_relpath=f"reflectance.{key}",
                )
                if self.id is not None
                else None,
            )

        return result
