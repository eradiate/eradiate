import os
from typing import MutableMapping, Optional

import attr
import pint

from ..core import SceneElement
from ..spectra import Spectrum, SpectrumFactory
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_kernel as uck


@parse_docs
@attr.s
class MeshTreeElement(SceneElement):
    """
    Container class for mesh based constituents of tree-like objects in a canopy.
    Holds the filepath for the triangulated mesh and all parameters specifying
    the associated BSDF.

    Note: The triangulated mesh must be provided in .ply or .obj format.

    Since mesh definition files cannot carry Pint units, the attribute ``mesh_unit``
    lets users provide the unit which their mesh is defined in. Upon kernel_dict
    creation the mesh is scaled to match the length unit used in the kernel.
    If ``mesh_unit` is not provided, no scaling of the mesh is performed.

    The :meth:`MeshTreeElement.from_dict` constructor instantiates the class from
    a configuration dictionary.
    """

    id = documented(
        attr.ib(
            default="mesh_tree_element",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default="'mesh_tree_element'",
    )

    mesh_filename = documented(
        attr.ib(
            default=None,
            validator=validators.is_path,
        ),
        doc="Path to the triangulated mesh data file.",
        type="path-like",
        default="None",
    )

    @mesh_filename.validator
    def _mesh_filename_validator(self, attribute, value):
        file_extension = os.path.splitext(value)[1]

        if not file_extension in [".obj", ".ply"]:
            raise ValueError(
                f"While validating {attribute.name}: File extension must be '.obj'"
                f"or '.ply', got {file_extension}"
            )

    mesh_unit = documented(
        attr.ib(
            default=None,
        ),
        doc="Unit the mesh was defined in. Used to conver to kernel units",
        type="pint.Unit or None",
        default="None",
    )

    @mesh_unit.validator
    def _mesh_unit_validator(self, attribute, value):
        if not isinstance(value, pint.unit.Unit):
            raise ValueError(
                f"While validating {attribute.name}:Mesh unit parameter must be "
                f"a pint Unit object, got {type(value)}"
            )

    mesh_reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the mesh. "
        "Must be a reflectance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.5",
    )

    mesh_transmittance = documented(
        attr.ib(
            default=0.0,
            converter=SpectrumFactory.converter("transmittance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("transmittance"),
            ],
        ),
        doc="Transmittance spectrum of the mesh. "
        "Must be a transmittance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.0",
    )

    def shapes(self, ctx=None):
        """
        Return shape plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a :class:`.KernelDict`
            containing all the shapes in the leaf cloud.
        """
        from mitsuba.core import ScalarTransform4f

        if ctx.ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs(ctx=ctx)[f"bsdf_{self.id}"]

        if self.mesh_unit is None:
            scaling_factor = 1
        else:
            kernel_length = uck.get("length")
            scaling_factor = (1 * self.mesh_unit).m_as(kernel_length)

        file_extension = os.path.splitext(self.mesh_filename)[1]
        if file_extension == ".obj":
            shapes_dict = {
                self.id: {
                    "type": "obj",
                    "filename": self.mesh_filename,
                    "bsdf": bsdf,
                    "to_world": ScalarTransform4f.scale(scaling_factor),
                }
            }
        elif file_extension == ".ply":
            shapes_dict = {
                self.id: {
                    "type": "ply",
                    "filename": self.mesh_filename,
                    "bsdf": bsdf,
                    "to_world": ScalarTransform4f.scale(scaling_factor),
                }
            }

        return shapes_dict

    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a :class:`.KernelDict`
            containing all the BSDFs attached to the shapes in the leaf cloud.
        """
        return {
            f"bsdf_{self.id}": {
                "type": "bilambertian",
                "reflectance": self.mesh_reflectance.kernel_dict(ctx=ctx)["spectrum"],
                "transmittance": self.mesh_transmittance.kernel_dict(ctx=ctx)[
                    "spectrum"
                ],
            }
        }

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if not ctx.ref:
            return self.shapes(ctx=ctx)
        else:
            return {**self.bsdfs(ctx=ctx), **self.shapes(ctx=ctx)}
