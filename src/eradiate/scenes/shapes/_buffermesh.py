from __future__ import annotations

import warnings

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._core import ShapeInstance
from ..core import BoundingBox, traverse
from ...attrs import define, documented
from ...contexts import KernelContext
from ...kernel import SceneParameter
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck


@define(eq=False, slots=False)
class BufferMeshShape(ShapeInstance):
    """
    Buffer mesh shape [``buffer_mesh``].

    This shape represents a triangulated mesh directly defined by lists of
    vertex coordinates and face definitions.

    Notes
    -----
    * This class does not support the ``to_world`` parameter: mesh vertices
      must be transformed manually.

    * The optional ``texcoords`` field can hold a list of (u, v) texture
      coordinates in the same order as vertex coordinates.
    """

    vertices: pint.Quantity = documented(
        pinttr.field(
            validator=pinttr.validators.has_compatible_units,
            units=ucc.deferred("length"),
            kw_only=True,
        ),
        doc="List of vertex coordinates, specified either as a (n, 3) NumPy "
        "array or a list of triplets.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
    )

    faces: np.ndarray = documented(
        attrs.field(kw_only=True, converter=np.array),
        doc="List of face definitions, specified either as a (n, 3) NumPy "
        "array or a list of triplets of vertex indices.",
        type="ndarray",
        init_type="array-like",
    )

    texcoords: np.ndarray | None = documented(
        attrs.field(
            kw_only=True, default=None, converter=attrs.converters.optional(np.array)
        ),
        doc="List of vertex texture coordinates, specified either as a (n, 2) "
        "NumPy array or a list of pairs.",
        type="ndarray",
        init_type="array-like",
    )

    @vertices.validator
    @faces.validator
    def _vertex_face_validator(self, attribute, value):
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 3:
            raise ValueError(
                f"while validating {attribute.name}, must be an array of shape "
                f"(n, 3), got {value.shape}"
            )

    @texcoords.validator
    def _texcoords_validator(self, attribute, value):
        if value is None:
            return

        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 2:
            raise ValueError(
                f"while validating {attribute.name}, must be an array of shape "
                f"(n, 2), got {value.shape}"
            )

    def __attrs_post_init__(self):
        if self.to_world is not None:
            warnings.warn(
                "Setting the 'to_world' field of a BufferMeshShape instance is "
                "not supported: it will be ignored."
            )

        self.update()

    def bbox(self) -> BoundingBox:
        # Inherit docstring
        raise NotImplementedError

    @property
    def instance(self) -> mi.Object:
        if self.bsdf is not None:
            template, _ = traverse(self.bsdf)
            kdict = template.render(ctx=KernelContext())
            kdict["id"] = self._bsdf_id  # TODO: Enforce ID control at BSDF level
            bsdf = mi.load_dict(kdict)
        else:
            bsdf = None

        props = mi.Properties()
        props["mesh_bsdf"] = bsdf

        mesh = mi.Mesh(
            name=self.id,
            face_count=self.faces.shape[0],
            vertex_count=self.vertices.shape[0],
            has_vertex_normals=False,
            has_vertex_texcoords=self.texcoords is not None,
            props=props,
        )

        vertices = self.vertices.m_as(uck.get("length"))
        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = vertices.ravel()
        mesh_params["faces"] = self.faces.ravel()

        if self.texcoords is not None:
            mesh_params["vertex_texcoords"] = self.texcoords.ravel()

        mesh_params.update()

        return mesh

    def write_ply(self, filename: str) -> None:
        """
        Write the mesh data to a PLY file.

        Parameters
        ----------
        filename : str
            Path and filename to write the mesh file into. No directories
            are created.

        Notes
        -----
        Vertex coordinates are expressed in kernel units and accordingly prior
        to writing to disk. See the documentation of
        :data:`eradiate.unit_context_kernel` and the
        :ref:`sec-user_guide-unit_guide_user`.
        """
        self.instance.write_ply(filename)

    @property
    def params(self) -> dict[str, SceneParameter] | None:
        if self.bsdf is None:
            return None

        _, params = traverse(attrs.evolve(self.bsdf, id=self._bsdf_id))
        return {f"bsdf.{k}": v for k, v in params.items()}
