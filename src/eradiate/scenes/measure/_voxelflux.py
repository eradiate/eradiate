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
class VoxelFluxMeasure(Measure):
    """
    Voxel flux measure scene element [``voxelflux``].

    This scene element creates a measure that records the flux traversing voxel
    faces in a 6D tensor structure with dimensions [D, F, X, Y, Z, C] where:
    - D: flux direction relative to axis (0 for negative, 1 for positive)
    - F: axis of the faces (0, 1, 2 for x, y, z)
    - X, Y, Z: indices of the face
    - C: channels
    The grid boundary is defined by the `.bounding_box` and the number of voxel
    along each axis by `.voxel_resolution`.

    Notes
    -----
    To count interactions over the  entire scene, leave `.bounding_box` set to None.
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
        doc="Outer boundary of the voxel flux grid. The `.voxel_resolution` "
        "field divides this bounding box along each axis in a regular grid. "
        "When set to None, the default behaviour is to use the scene's bounding "
        "box.",
        type=":class:`.BoundingBox` or None",
        init_type=":class:`.BoundingBox`, dict, tuple, or array-like, optional",
        default=None,
    )

    surface_flux: bool = documented(
        attrs.field(
            default=False,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="If ``True``, the foreshortening factor is multiplied to the accumulated flux.",
        type="bool",
        init_type="bool",
        default="False",
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
    def film_resolution(self) -> tuple[int, int, int, int, int]:
        """
        Returns the 5D tensor dimensions [D, F, X, Y, Z].
        """
        return (
            2,  # D: direction (0=negative, 1=positive)
            3,  # F: face axis (0=x, 1=y, 2=z)
            self.voxel_resolution[0] + 1,  # X: x-face indices
            self.voxel_resolution[1] + 1,  # Y: y-face indices
            self.voxel_resolution[2] + 1,  # Z: z-face indices
            1,  # C: channels
        )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        return "voxelflux"

    @property
    def template(self) -> dict:
        result = {
            "type": self.kernel_type,
            "id": self.sensor_id,
            "film.type": "tensorfilm",
            "film.ndims": 6,
            "film.sizes": f"2, 3, {self.voxel_resolution[0] + 1}, {self.voxel_resolution[1] + 1}, {self.voxel_resolution[2] + 1}, 1",
            "res_x": self.voxel_resolution[0],
            "res_y": self.voxel_resolution[1],
            "res_z": self.voxel_resolution[2],
            "surface_flux": self.surface_flux,
            "apply_sample_scale": self.apply_sample_scale,
        }

        if self.bounding_box:
            result["bbox_min"] = self.bounding_box.min.m_as(uck.get("length"))
            result["bbox_max"] = self.bounding_box.max.m_as(uck.get("length"))
        return result

    @property
    def var(self) -> tuple[str, dict]:
        # Inherit docstring
        return "flux", {
            "standard_name": "flux",
            "long_name": "flux",
            "units": symbol(symbol(ureg.watt)),
        }

    @property
    def tensor_to_dataarray(self) -> dict:
        return {
            "direction": ("direction", ["negative", "positive"]),
            "face": ("face", ["x", "y", "z"]),
            "x_index": ("x_index", range(self.voxel_resolution[0] + 1)),
            "y_index": ("y_index", range(self.voxel_resolution[1] + 1)),
            "z_index": ("z_index", range(self.voxel_resolution[2] + 1)),
            "channel": ("channel", range(1)),
        }
