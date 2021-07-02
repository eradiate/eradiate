import os
from pathlib import Path
from typing import MutableMapping, Optional

import attr
import pint

from ..core import SceneElement
from ..spectra import Spectrum, SpectrumFactory
from ... import validators
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@parse_docs
@attr.s
class MeshTreeElement(SceneElement):
    """
    Container class for mesh based constituents of tree-like objects in a canopy.
    Holds the filepath for the triangulated mesh and all parameters specifying
    the associated BSDF.

    .. important:: The triangulated mesh must be provided in .ply or .obj format.

    Since mesh definition files cannot carry Pint units, the attribute ``mesh_units``
    lets users provide the unit which their mesh is defined in. Upon kernel dict
    creation the mesh is scaled to match the length unit used in the kernel.
    If ``mesh_units` is not provided, no scaling of the mesh is performed.

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
            converter=attr.converters.optional(Path),
            default=None,
        ),
        doc="Path to the triangulated mesh data file. This parameter is required.",
        type="path-like",
    )

    @mesh_filename.validator
    def _mesh_filename_validator(self, attribute, value):
        if value is None:
            raise ValueError("'mesh_filename' is required")

        validators.path_exists(self, attribute, value)

        if not value.suffix in [".obj", ".ply"]:
            raise ValueError(
                f"While validating {attribute.name}: File extension must be '.obj'"
                f"or '.ply', got {value.suffix}"
            )

    mesh_units = documented(
        attr.ib(default=None, converter=attr.converters.optional(ureg.Unit)),
        doc="Units the mesh was defined in. Used to convert to kernel units. "
        "If this value is ``None``, the mesh is interpreted as being defined in"
        "kernel units.",
        type="str or :class:`pint.Unit` or None",
        default="None",
    )

    @mesh_units.validator
    def _mesh_units_validator(self, attribute, value):
        if not isinstance(value, pint.unit.Unit):
            raise ValueError(
                f"While validating {attribute.name}: Mesh unit parameter must be "
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
        from mitsuba.core import ScalarTransform4f

        if ctx.ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs(ctx=ctx)[f"bsdf_{self.id}"]

        if self.mesh_units is None:
            scaling_factor = 1.0
        else:
            kernel_length = uck.get("length")
            scaling_factor = (1.0 * self.mesh_units).m_as(kernel_length)

        base_dict = {
            "filename": str(self.mesh_filename),
            "bsdf": bsdf,
            "to_world": ScalarTransform4f.scale(scaling_factor),
        }

        if self.mesh_filename.suffix == ".obj":
            base_dict["type"] = "obj"
        elif self.mesh_filename.suffix == ".ply":
            base_dict["type"] = "ply"
        else:
            raise ValueError(
                f"unsupported file extension '{self.mesh_filename.suffix}'"
            )

        return {self.id: base_dict}

    def bsdfs(self, ctx=None):
        return {
            f"bsdf_{self.id}": {
                "type": "bilambertian",
                "reflectance": self.mesh_reflectance.kernel_dict(ctx=ctx)["spectrum"],
                "transmittance": self.mesh_transmittance.kernel_dict(ctx=ctx)[
                    "spectrum"
                ],
            }
        }

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create an :class:`.MeshTreeElement`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return MeshTreeElement.from_dict(value)

        return value

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if not ctx.ref:
            return self.shapes(ctx=ctx)
        else:
            return {**self.bsdfs(ctx=ctx), **self.shapes(ctx=ctx)}
