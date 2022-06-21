from __future__ import annotations

from pathlib import Path

import attr

from ._core import Shape, shape_factory
from ..core import KernelDict
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...util.misc import onedict_value


@parse_docs
@attr.s
class FileMeshShape(Shape):
    """
    File based mesh shape [``file_mesh``].

    This shape represents a triangulated mesh defined in a file
    in either the PLY or OBJ format.
    The vertex positions are assumed to be defined in kernel units.
    """

    filename: Path = documented(
        attr.ib(converter=Path, kw_only=True),
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

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        if self.filename.suffix == ".obj":
            type = "obj"
        elif self.filename.suffix == ".ply":
            type = "ply"
        else:
            raise ValueError(f"unknown mesh file type '{self.filename.suffix}'")

        meshdict = {
            "type": type,
            "filename": str(self.filename),
            "face_normals": True,
        }

        if self.bsdf:
            meshdict["bsdf"] = onedict_value(self.bsdf.kernel_dict(ctx=ctx))

        result = KernelDict({self.id: meshdict})

        return result
