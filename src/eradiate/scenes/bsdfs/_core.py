from __future__ import annotations

from abc import ABC

import attrs

from ..core import NodeSceneElement
from ..filters import FilterType
from ..._factory import Factory
from ...attrs import define, documented

bsdf_factory = Factory()
bsdf_factory.register_lazy_batch(
    [
        ("_black.BlackBSDF", "black", {}),
        ("_checkerboard.CheckerboardBSDF", "checkerboard", {}),
        ("_lambertian.LambertianBSDF", "lambertian", {}),
        ("_mqdiffuse.MQDiffuseBSDF", "mqdiffuse", {}),
        ("_ocean_legacy.OceanLegacyBSDF", "ocean_legacy", {}),
        ("_ocean_grasp.OceanGraspBSDF", "ocean_grasp", {}),
        ("_ocean_mishchenko.OceanMishchenkoBSDF", "ocean_mishchenko", {}),
        ("_opacity_mask.OpacityMaskBSDF", "opacity_mask", {}),
        ("_rpv.RPVBSDF", "rpv", {}),
        ("_rtls.RTLSBSDF", "rtls", {}),
        ("_hapke.HapkeBSDF", "hapke", {}),
    ],
    cls_prefix="eradiate.scenes.bsdfs",
)


@define(eq=False, slots=False)
class BSDF(NodeSceneElement, ABC):
    """
    Abstract base class for all BSDF scene elements.
    """

    filter_type: FilterType = documented(
        attrs.field(
            default=FilterType.INCLUDE,
            converter=FilterType,
            validator=attrs.validators.instance_of(FilterType),
        ),
        doc="Filter type controlling whether interactions with this BSDF "
        "should be included in sensor measurements.",
        type=".FilterType",
        init_type=".FilterType or int",
        default="FilterType.INCLUDE",
    )

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        """Return the plugin name used to instantiate the kernel counterpart."""
        raise NotImplementedError

    @property
    def template(self) -> dict:
        """Build base template with filter parameter."""
        result = {
            "type": self.kernel_type,
            "filter": int(self.filter_type),
        }

        if self.id is not None:
            result["id"] = self.id

        return result
