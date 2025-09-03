from __future__ import annotations

import attrs
import pinttr

from ._core import Measure
from ..core import BoundingBox
from ... import validators
from ...attrs import define, documented
from ...units import symbol
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@define(eq=False, slots=False)
class AbsorbedFluxMeasure(Measure):
    """
    Absorbed flux measure scene element [``absorbedflux``, ``absorbedflux``].

    This scene element creates a measure that records the flux absorbed by
    surface interactions in a 3D grid. The grid boundary is defined by the
    `.bounding_box` and the number of voxel along each axis by `.voxel_resolution`.
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
        doc="A 3-vector specifying the number of voxels along each axis. "
        "This vector must contain positive integers.",
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

    bounding_box: BoundingBox | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(BoundingBox.convert),
        ),
        doc="Outer boundary of the voxel grid. The `.voxel_resolution` "
        "field divides this bounding box along each axis in a regular grid. "
        "When set to None, the default behaviour is to use the scene's bounding "
        "box.",
        type=":class:`.BoundingBox` or None",
        init_type=":class:`.BoundingBox`, dict, tuple, or array-like, optional",
        default=None,
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
        return "absorbedflux"

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

        if self.bounding_box:
            result["bbox_min"] = self.bounding_box.min.m_as(uck.get("length"))
            result["bbox_max"] = self.bounding_box.max.m_as(uck.get("length"))
        return result

    @property
    def var(self) -> tuple[str, dict]:
        # Inherit docstring
        return "absorbed_flux", {
            "standard_name": "absorbed_flux",
            "long_name": "absorbed flux",
            "units": symbol(ureg.watt),
        }

    @property
    def tensor_to_dataarray(self) -> dict:
        return {
            "x_index": ("x_index", range(self.voxel_resolution[0])),
            "y_index": ("y_index", range(self.voxel_resolution[1])),
            "z_index": ("z_index", range(self.voxel_resolution[2])),
            "channel": ("channel", range(1)),
        }
