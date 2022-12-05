from __future__ import annotations

from pathlib import Path

import attrs

from ._core import Shape
from ..core import NodeSceneElement
from ...attrs import documented, parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class FileMeshShape(Shape, NodeSceneElement):
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
    def kernel_type(self) -> str:
        if self.filename.suffix == ".obj":
            return "obj"
        elif self.filename.suffix == ".ply":
            return "ply"
        else:
            raise ValueError(f"unknown mesh file type '{self.filename.suffix}'")

    @property
    def template(self) -> dict:
        return {
            **super().template,
            "filename": str(self.filename),
            "face_normals": True,
        }
