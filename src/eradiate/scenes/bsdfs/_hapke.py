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
class HapkeBSDF(BSDF):
    """
    Hapke BSDF [``hapke``].

    This BSDF implements the Hapke surface model as described in
    :cite:`Hapke1984BidirectionalReflectanceSpectroscopy`. This highly flexible
    and robust surface model allows for the characterisation of a sharp
    back-scattering hot spot. The so-called Hapke model has been adapted to
    several different use cases in the litterature, the version with 6
    parameters implemented here is one of the most commonly used.

    See Also
    --------
    :ref:`plugin-bsdf-hapke`
    """

    w: Spectrum = documented(
        attrs.field(
            default=None,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Single scattering albedo 'w'. Must be in [0; 1]",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
    )

    b: Spectrum = documented(
        attrs.field(
            default=None,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Anisotropy parameter 'b' Must be in [0; 1]",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
    )

    c: Spectrum | None = documented(
        attrs.field(
            default=None,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Scattering coefficient 'c'. Must be in [0; 1]",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
    )

    theta: Spectrum = documented(
        attrs.field(
            default=0.183,
            converter=spectrum_factory.converter("angle"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("angle"),
            ],
        ),
        doc="Photometric roughness 'theta'. Angle in degree. Must be in [0; 90]°",
        type=".Spectrum",
        init_type="quantity or float",
    )

    B_0: Spectrum = documented(
        attrs.field(
            default=None,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Shadow hiding opposition effect amplitude 'B_0'. Must be in [0; 1]",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
    )

    h: Spectrum = documented(
        attrs.field(
            default=None,
            converter=spectrum_factory.converter("dimensionless"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="shadow hiding opposition effect width 'h'. Must be in [0; 1]",
        type=".Spectrum",
        init_type=".Spectrum or dict or float",
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
