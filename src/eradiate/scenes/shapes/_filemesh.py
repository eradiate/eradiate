from __future__ import annotations

from pathlib import Path

import attrs

from ._core import ShapeNode
from ..core import BoundingBox
from ...attrs import documented, parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class FileMeshShape(ShapeNode):
    """
    File based mesh shape [``file_mesh``].

    This shape represents a triangulated mesh defined in a file
    in either the PLY or OBJ format.
    The vertex positions are assumed to be defined in kernel units.
    """

    filename: Path = documented(
        attrs.field(converter=Path, kw_only=True),
        type="Path",
        init_type="path-like",
        doc="Path to the mesh file.",
    )

    @filename.validator
    def _filename_validator(self, attribute, value):
        if value.suffix not in [".obj", ".ply"]:
            raise ValueError(
                f"while validating {attribute.name}:"
                f"Eradiate supports mesh files only in PLY or OBJ format."
            )

    @property
    def _kernel_type(self) -> str:
        if self.filename.suffix == ".obj":
            return "obj"
        elif self.filename.suffix == ".ply":
            return "ply"
        else:
            raise ValueError(f"unknown mesh file type '{self.filename.suffix}'")

    def bbox(self) -> BoundingBox:
        # Inherit docstring
        raise NotImplementedError

    @property
    def template(self) -> dict:
        # Inherit docstring
        return {
            "type": self._kernel_type,
            "filename": str(self.filename),
            "face_normals": True,
        }
