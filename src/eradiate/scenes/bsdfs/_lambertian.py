from __future__ import annotations

import attrs
import mitsuba as mi

from ._core import BSDF
from ..core import traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...kernel import SceneParameter, SearchSceneParameter


@define(eq=False, slots=False)
class LambertianBSDF(BSDF):
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
        doc="Reflectance spectrum. Can be initialized with a dictionary "
        "processed by :data:`.spectrum_factory`.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.5",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = {
            "type": "diffuse",
            **{
                f"reflectance.{key}": value
                for key, value in traverse(self.reflectance)[0].items()
            },
        }

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        params = traverse(self.reflectance)[1].data

        result = {}
        for key, param in params.items():
            result[f"reflectance.{key}"] = attrs.evolve(
                param,
                search=SearchSceneParameter(
                    node_type=mi.BSDF,
                    node_id=self.id,
                    parameter_relpath=f"reflectance.{key}",
                )
                if self.id is not None
                else None,
            )

        return result
