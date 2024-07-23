from __future__ import annotations

from pathlib import Path

import attrs

from ._core import ShapeNode
from ..core import BoundingBox
from ...attrs import define, documented


@define(eq=False, slots=False)
class FileMeshShape(ShapeNode):
    """
    File based mesh shape [``file_mesh``].

    This shape represents a triangulated mesh defined in a file. The OBJ and PLY
    formats are supported.

    Warnings
    --------
    Vertex coordinates are assumed to be defined in kernel units.
    """

    filename: Path = documented(
        attrs.field(converter=Path, kw_only=True),
        type="Path",
        init_type="path-like",
        doc="Path to the mesh file.",
    )

    @filename.validator
    def _filename_validator(self, attribute, value):
        if value.suffix not in {".obj", ".ply"}:
            raise ValueError(
                f"while validating {attribute.name}:"
                f"Eradiate supports mesh files only in PLY or OBJ format."
            )

    def bbox(self) -> BoundingBox:
        # Inherit docstring
        raise NotImplementedError

    @property
    def template(self) -> dict:
        # Inherit docstring

        if self.filename.suffix == ".obj":
            mi_plugin = "obj"
        elif self.filename.suffix == ".ply":
            mi_plugin = "ply"
        else:
            raise ValueError(
                f"unsupported mesh file extension '{self.filename.suffix}'"
            )
        result = {
            "type": mi_plugin,
            "filename": str(self.filename),
            "face_normals": True,
        }
        if self.to_world is not None:
            result["to_world"] = self.to_world

        return result
