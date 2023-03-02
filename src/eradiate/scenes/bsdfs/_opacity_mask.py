from __future__ import annotations

import attrs
import mitsuba as mi

from ._core import BSDF, bsdf_factory
from ._lambertian import LambertianBSDF
from ..core import traverse
from ...attrs import documented, parse_docs
from ...kernel import TypeIdLookupStrategy, UpdateParameter


@parse_docs
@attrs.define(eq=False, slots=False)
class OpacityMaskBSDF(BSDF):
    """
    Opacity Mask BSDF [``opacity_mask``]
    """

    opacity_bitmap: "mi.Bitmap" = documented(
        attrs.field(kw_only=True),
        doc="Mitsuba bitmap that specifies the opacity of the nested BSDF plugin",
        type="mitsuba.Bitmap",
        init_type="mitsuba.Bitmap or dict",
    )

    @opacity_bitmap.validator
    def _opacity_bitmap_validator(self, attribute, value):
        if value is not None:
            if not isinstance(value, mi.Bitmap):
                raise TypeError(
                    f"while validating '{attribute.name}': "
                    f"'{attribute.name}' must be a mitsuba Bitmap instance;"
                    f"found: {type(value)}",
                )

    uv_trafo: "mi.ScalarTransform4f" = documented(
        attrs.field(default=None),
        doc="Transform to scale the opacity mask.",
        type="mitsuba.ScalarTransform4f",
        init_type="mitsuba.ScalarTransform4f or dict",
        default="None",
    )

    @uv_trafo.validator
    def _uv_trafo_validator(self, attribute, value):
        if value is not None:
            if not isinstance(value, mi.ScalarTransform4f):
                raise TypeError(
                    f"while validating '{attribute.name}': "
                    f"'{attribute.name}' must be a mitsuba ScalarTransform4f instance;"
                    f"found: {type(value)}"
                )

    nested_bsdf: BSDF = documented(
        attrs.field(
            factory=LambertianBSDF,
            converter=bsdf_factory.convert,
            validator=attrs.validators.instance_of(BSDF),
        ),
        doc="The reflection model attached to the surface.",
        type=".BSDF",
        init_type=".BSDF or dict, optional",
        default=":class:`LambertianBSDF() <.LambertianBSDF>`",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring

        result = {
            "type": "mask",
            "id": self.id,
            "opacity.type": "bitmap",
            "opacity.bitmap": self.opacity_bitmap,
            "opacity.filter_type": "nearest",
            "opacity.wrap_mode": "clamp",
        }

        if self.uv_trafo is not None:
            result["opacity.to_uv"] = self.uv_trafo

        for key, value in traverse(self.nested_bsdf)[0].items():
            result[f"nested_bsdf.{key}"] = value

        return result

    @property
    def params(self) -> dict[str, UpdateParameter]:
        # Inherit Docstring

        result = {}

        # improve this, by using the BSDFs own lookup strategy
        for key, param in traverse(self.nested_bsdf)[1].items():
            result[f"nested_bsdf.{key}"] = attrs.evolve(
                param,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF,
                    node_id=self.id,
                    parameter_relpath=f"nested_bsdf.{key}",
                )
                if self.id is not None
                else None,
            )

        return result
