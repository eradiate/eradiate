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
class MaignanBSDF(BSDF):
    """
    Maignan polarized BSDF [``maignan``].

    This plugin implements the reflection model proposed by
    :cite:`Maignan2009PolarizedReflectance`. The model is based on fits to
    POLDER observations and combines a BRDF with the Fresnel
    reflectance matrix (see Eq. 21 in :cite:`Maignan2009PolarizedReflectance`).
    """

    C: Spectrum = documented(
        attrs.field(
            default=5.0,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Scaling parameter (*C*) of the Maignan BPDF model. Must be "
        "positive (typically in [4, 8].)",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="5.0",
    )

    ndvi: Spectrum = documented(
        attrs.field(
            default=0.8,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="NDVI parameter (*ν*) of the Maignan BPDF model. Must be "
        "in :math:`[0, 1]`.)",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="1.0",
    )

    refr_re: Spectrum = documented(
        attrs.field(
            default=1.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Real part of the surface's refractive index. Must be positive.)",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="1.5",
    )

    refr_im: Spectrum = documented(
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Imaginary part of the surface's refractive index. Must be positive.)",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="0.0",
    )

    ext_ior: Spectrum = documented(
        attrs.field(
            default=1.000277,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Exterior index of refraction. The imaginary part is assumed to be 0.",
        type=".Spectrum",
        init_type=".Spectrum or dict or float, optional",
        default="1.000277",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = {"type": "maignan"}

        for obj_name in ["C", "ndvi", "refr_re", "refr_im", "ext_ior"]:
            kdict = traverse(getattr(self, obj_name))[0]
            for key, value in kdict.items():
                result[f"{obj_name}.{key}"] = value

        if self.id is not None:
            result["id"] = self.id

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        result = {}

        for obj_name in ["C", "ndvi", "refr_re", "refr_im", "ext_ior"]:
            params = traverse(getattr(self, obj_name))[1].data

            for key, param in params.items():
                result[f"{obj_name}.{key}"] = attrs.evolve(
                    param,
                    search=SearchSceneParameter(
                        node_type=mi.BSDF,
                        node_id=self.id,
                        parameter_relpath=f"{obj_name}.{key}",
                    )
                    if self.id is not None
                    else None,
                )

        return result
