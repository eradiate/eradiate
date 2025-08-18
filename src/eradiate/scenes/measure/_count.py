from __future__ import annotations

import attrs
import pinttr

from ._core import Measure
from ..core import BoundingBox
from ... import validators
from ...attrs import define, documented
from ...units import unit_context_kernel as uck


@define(eq=False, slots=False)
class CountMeasure(Measure):
    """
    Count measure scene element [``count``, ``count``].

    This scene element creates a measure that counts the number of interactions.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------
    voxel_resolution: tuple[int, int, int] = documented(
        pinttr.field(
            default=(1, 1, 1),
            converter=tuple,
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(int),
                iterable_validator=validators.has_len(3),
            ),
        ),
        doc="A 3-vector specifying the up direction of the spot.\n"
        "This vector must be different from the spots's pointing direction,\n"
        "which is given by ``target - origin``.",
        type="array",
        init_type="array-like",
        default="[1, 1, 1]",
    )

    @voxel_resolution.validator
    def _voxel_resolution_validator(self, attribute, value):
        if value is not None:
            if (value[0] <= 0) or (value[1] <= 0) or (value[2] <= 0):
                raise ValueError(
                    f"While initializing {attribute}: "
                    f"Voxel resolution must be positive, got {value}"
                )

    boundary: BoundingBox | None = documented(
        attrs.field(
            default=None,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(BoundingBox)
            ),
            # converter=attrs.converters.optional(BoundingBox),
        ),
        doc="Bounding box of the measure.",
    )

    apply_sample_scale: bool = documented(
        attrs.field(
            default=True,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="If ``True``, the sample scale will be applied to the measure.",
        type="bool",
        init_type="bool",
        default="True",
    )

    @property
    def film_resolution(self) -> tuple[int, int, int]:
        return (
            self.voxel_resolution[0],
            self.voxel_resolution[1],
            self.voxel_resolution[2],
        )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        return "count"

    @property
    def template(self) -> dict:
        result = {
            "type": self.kernel_type,
            "id": self.sensor_id,
            "film.type": "volfilm",
            "film.res_x": self.voxel_resolution[0],
            "film.res_y": self.voxel_resolution[1],
            "film.res_z": self.voxel_resolution[2],
            "apply_sample_scale": self.apply_sample_scale,
        }

        if self.boundary:
            result["bbox_min"] = self.boundary.min.m_as(uck.get("length"))
            result["bbox_max"] = self.boundary.max.m_as(uck.get("length"))
        return result
