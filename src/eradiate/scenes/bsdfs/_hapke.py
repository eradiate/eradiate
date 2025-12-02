from __future__ import annotations

import attrs
import mitsuba as mi

from ._core import BSDF
from ..core import traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...kernel import SceneParameter, SearchSceneParameter
from ...units import unit_registry as ureg


@define(eq=False, slots=False)
class HapkeBSDF(BSDF):
    """
    Hapke BSDF [``hapke``].

    This BSDF implements  a bare soil reflection model based on the work of
    Bruce Hapke. This variant is validated against the one presented by
    :cite:t:`Nguyen2025MappingSurfaceProperties`.
    It features 6 parameters and includes adjustments compared to the core
    reference :cite:p:`Hapke2012TheoryReflectanceEmittance`. The unit test suite
    used to validate this implementation used reference data from
    :cite:t:`Pommerol2013PhotometricPropertiesMars`.

    The default parameters are an order of magnitude of the results presented by
    :cite:t:`Nguyen2025MappingSurfaceProperties` and notably neglect the
    influence of the opposition effect.

    See Also
    --------
    :ref:`plugin-bsdf-hapke`
    """

    w: Spectrum = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Single scattering albedo ω. Must be in [0, 1].",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.5",
    )

    b: Spectrum = documented(
        attrs.field(
            default=0.2,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Asymmetry parameter of the Henyey-Greenstein phase function. "
        "Must be in [0, 1].",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.2",
    )

    c: Spectrum | None = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Backscattering parameter of the Henyey-Greenstein phase function. "
        "Must be in [0, 1].",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.5",
    )

    theta: Spectrum = documented(
        attrs.field(
            default=30.0 * ureg.deg,
            converter=spectrum_factory.converter("angle"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("angle"),
            ],
        ),
        doc="Photometric roughness θ. Angle in degree. Must be in [0, 90]°.",
        type=".Spectrum",
        init_type="quantity or float",
        default="30.0",
    )

    B_0: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Intensity of shadow hiding opposition effect. Must be in [0, 1].",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.0",
    )

    h: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Width of shadow hiding opposition effect . Must be in [0, 1].",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
        default="0.0",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring
        objects = {
            "w": traverse(self.w)[0],
            "b": traverse(self.b)[0],
            "c": traverse(self.c)[0],
            "theta": traverse(self.theta)[0],
            "B_0": traverse(self.B_0)[0],
            "h": traverse(self.h)[0],
        }

        result = {"type": "hapke"}

        for obj_key, obj_values in objects.items():
            for key, value in obj_values.items():
                result[f"{obj_key}.{key}"] = value

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        objects = {
            "w": traverse(self.w)[1],
            "b": traverse(self.b)[1],
            "c": traverse(self.c)[1],
            "theta": traverse(self.theta)[1],
            "B_0": traverse(self.B_0)[1],
            "h": traverse(self.h)[1],
        }

        result = {}
        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{obj_key}.{key}"] = attrs.evolve(
                    param,
                    search=SearchSceneParameter(
                        node_type=mi.BSDF,
                        node_id=self.id,
                        parameter_relpath=f"{obj_key}.{key}",
                    )
                    if self.id is not None
                    else None,
                )

        return result
