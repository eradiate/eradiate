from __future__ import annotations

import attrs
import mitsuba as mi
import numpy as np

from ._core import BSDF, bsdf_factory
from ._lambertian import LambertianBSDF
from ..core import traverse
from ... import converters
from ...attrs import define, documented
from ...kernel import SceneParameter, SearchSceneParameter


def _to_bitmap(value):
    if isinstance(value, mi.Bitmap):
        return value

    elif isinstance(value, np.ndarray):
        return mi.Bitmap(value)

    elif isinstance(value, list):
        return mi.Bitmap(np.array(value))

    else:
        return value


@define(eq=False, slots=False)
class OpacityMaskBSDF(BSDF):
    """
    Opacity Mask BSDF [``opacity_mask``]
    """

    opacity_bitmap: np.typing.ArrayLike | "mi.Bitmap" = documented(
        attrs.field(converter=_to_bitmap, kw_only=True),
        doc="Mitsuba bitmap that specifies the opacity of the nested BSDF "
        "plugin. This parameter has no default and is required.",
        init_type="array-like or mitsuba.Bitmap",
        type="mitsuba.Bitmap",
    )

    @opacity_bitmap.validator
    def _opacity_bitmap_validator(self, attribute, value):
        if value is not None:
            if not isinstance(value, mi.Bitmap):
                raise TypeError(
                    f"while validating '{attribute.name}': "
                    f"'{attribute.name}' must be a mitsuba Bitmap instance; "
                    f"found: {type(value)}",
                )

    uv_trafo: "mi.ScalarTransform4f" = documented(
        attrs.field(converter=converters.to_mi_scalar_transform, kw_only=True),
        doc="Transform to scale the opacity mask. This parameter has no "
        "default and is required.",
        init_type="array-like or mitsuba.ScalarTransform4f",
        type="mitsuba.ScalarTransform4f",
    )

    @uv_trafo.validator
    def _uv_trafo_validator(self, attribute, value):
        if value is not None:
            if not isinstance(value, mi.ScalarTransform4f):
                raise TypeError(
                    f"while validating '{attribute.name}': "
                    f"'{attribute.name}' must be a mitsuba ScalarTransform4f instance; "
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
            "opacity.type": "bitmap",
            "opacity.bitmap": self.opacity_bitmap,
            "opacity.filter_type": "nearest",
            "opacity.wrap_mode": "clamp",
        }

        if self.id is not None:
            result["id"] = self.id

        if self.uv_trafo is not None:
            result["opacity.to_uv"] = self.uv_trafo

        for key, value in traverse(self.nested_bsdf)[0].items():
            result[f"nested_bsdf.{key}"] = value

        return result

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit Docstring

        result = {}

        # TODO: Improve this by using the BSDF's own lookup strategy
        for key, param in traverse(self.nested_bsdf)[1].items():
            result[f"nested_bsdf.{key}"] = attrs.evolve(
                param,
                search=SearchSceneParameter(
                    node_type=mi.BSDF,
                    node_id=self.id,
                    parameter_relpath=f"nested_bsdf.{key}",
                )
                if self.id is not None
                else None,
            )

        return result
