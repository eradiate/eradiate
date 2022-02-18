from __future__ import annotations

import typing as t
from abc import ABC
from collections.abc import MutableMapping
from pathlib import Path

import attr
import mitsuba as mi
import pint
import pinttr

from ._core import CanopyElement, biosphere_factory
from ._leaf_cloud import LeafCloud
from ..core import SceneElement
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@parse_docs
@attr.s
class Tree(CanopyElement, ABC):
    """
    Abstract base class for tree-like canopy elements.
    """

    pass


def _leaf_cloud_converter(value):
    """
    Special converter for the AbstractTree.leaf_cloud field.
    """
    if isinstance(value, MutableMapping):
        value["type"] = "leaf_cloud"
    return biosphere_factory.convert(value)


@biosphere_factory.register(type_id="abstract_tree")
@parse_docs
@attr.s
class AbstractTree(Tree):
    """
    A container class for abstract trees in discrete canopies.
    Holds a :class:`.LeafCloud` and the parameters characterizing a cylindrical
    trunk. The entire tree is described in local coordinates and can be placed
    in the scene using :class:`.InstancedCanopyElement`.

    The trunk starts at [0, 0, -0.1] and extends
    to [0, 0, trunk_height]. The trunk extends below ``z=0`` to avoid intersection
    issues at the intersection of the trunk and the ground the tree is usually placed on.

    The leaf cloud will by default be offset such that its local coordinate
    origin coincides with the upper end of the trunk. If this is not desired,
    e.g. the leaf cloud is centered around its coordinate origin and the trunk
    should not extend into it, the parameter ``leaf_cloud_extra_offset`` can be
    used to shift the leaf cloud **in addition** to the trunk's extent.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="abstract_tree",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"abstract_tree"',
    )

    leaf_cloud: t.Optional[LeafCloud] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(_leaf_cloud_converter),
            validator=attr.validators.optional(attr.validators.instance_of(LeafCloud)),
        ),
        doc="Instanced leaf cloud. Can be specified as a dictionary, which will "
        "be interpreted by :data:`.biosphere_factory`. If the latter case, the "
        '``"type"`` parameter, if omitted, will implicitly be set to '
        '``"leaf_cloud"``.',
        type=".LeafCloud or None",
        init_type=".LeafCloud or dict, optional",
    )

    trunk_height: pint.Quantity = documented(
        pinttr.ib(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc='Trunk height. Unit-enabled field (default: ucc["length"]).',
        type="quantity",
        init_type="quantity or float",
        default="1.0 m",
    )

    trunk_radius: pint.Quantity = documented(
        pinttr.ib(default=0.1 * ureg.m, units=ucc.deferred("length")),
        doc='Trunk radius. Unit-enabled field (default: ucc["length"]).',
        type="quantity",
        init_type="quantity or float, optional",
        default="0.1 m",
    )

    trunk_reflectance: Spectrum = documented(
        attr.ib(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the trunk. "
        "Must be a reflectance spectrum (dimensionless).",
        type=".Spectrum",
        init_type=".Spectrum or dict, optional",
        default="0.5",
    )

    leaf_cloud_extra_offset: pint.Quantity = documented(
        pinttr.ib(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Additional offset for the leaf cloud. 3-vector. "
        'Unit-enabled field (default: ucc["length"])',
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_bsdfs(self, ctx: KernelDictContext) -> t.Dict:
        """
        Return BSDF plugin specifications.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        dict
            Return a dictionary suitable for merge with a :class:`.KernelDict`
            containing all the BSDFs attached to the shapes
            in the abstract tree.
        """

        bsdfs_dict = self.leaf_cloud.kernel_bsdfs(ctx=ctx)

        bsdfs_dict[f"bsdf_{self.id}"] = {
            "type": "diffuse",
            "reflectance": self.trunk_reflectance.kernel_dict(ctx=ctx)["spectrum"],
        }

        return bsdfs_dict

    def kernel_shapes(self, ctx: KernelDictContext) -> t.Dict:
        """
        Return shape plugin specifications.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        dict
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the abstract tree.
        """
        kernel_length = uck.get("length")

        kernel_height = self.trunk_height.m_as(kernel_length)
        kernel_radius = self.trunk_radius.m_as(kernel_length)

        leaf_cloud = self.leaf_cloud.translated(
            [0.0, 0.0, kernel_height] * kernel_length
            + self.leaf_cloud_extra_offset.to(kernel_length)
        )

        if ctx.ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.kernel_bsdfs(ctx=ctx)[f"bsdf_{self.id}"]

        shapes_dict = leaf_cloud.kernel_shapes(ctx=ctx)

        shapes_dict[f"trunk_cyl_{self.id}"] = {
            "type": "cylinder",
            "bsdf": bsdf,
            "radius": kernel_radius,
            "p0": [0, 0, -0.1],
            "p1": [0, 0, kernel_height],
        }

        shapes_dict[f"trunk_cap_{self.id}"] = {
            "type": "disk",
            "bsdf": bsdf,
            "to_world": mi.ScalarTransform4f.scale(kernel_radius)
            * mi.ScalarTransform4f.translate([0, 0, kernel_height]),
        }

        return shapes_dict


@biosphere_factory.register(type_id="mesh_tree")
@parse_docs
@attr.s
class MeshTree(Tree):
    """
    A container class for mesh based tree-like objects in canopies.

    It holds one or more triangulated meshes and corresponding BSDFs, representing
    the tree.

    The mesh will be interpreted in local coordinates and should be used in an
    :class:`.InstancedCanopyElement` to place at arbitrary positions in a scene.
    """

    id: str = documented(
        attr.ib(
            default="mesh_tree",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"mesh_tree"',
    )

    mesh_tree_elements: t.List[MeshTree] = documented(
        attr.ib(
            factory=list,
            converter=lambda value: [
                MeshTreeElement.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [MeshTreeElement.convert(value)],
        ),
        doc="List of :class:`.CanopyElement` defining the canopy. Can be "
        "initialised with a :class:`.InstancedCanopyElement`, which will be "
        "automatically wrapped into a list. Dictionary-based specifications are "
        "allowed as well.",
        type="list of .InstancedCanopyElement",
        init_type="list of (.InstancedCanopyElement | dict)",
        default="[]",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_bsdfs(self, ctx: KernelDictContext) -> t.Dict:
        result = {}
        for mesh_tree_element in self.mesh_tree_elements:
            result = {**result, **mesh_tree_element.kernel_bsdfs(ctx=ctx)}
        return result

    def kernel_shapes(self, ctx: KernelDictContext) -> t.Dict:
        result = {}
        for mesh_tree_element in self.mesh_tree_elements:
            result = {**result, **mesh_tree_element.kernel_shapes(ctx=ctx)}
        return result


@parse_docs
@attr.s
class MeshTreeElement:
    """
    Container class for mesh based constituents of tree-like objects in a canopy.
    Holds the filepath for the triangulated mesh and all parameters specifying
    the associated BSDF.

    .. important:: The triangulated mesh must be provided in .ply or .obj format.

    Since mesh definition files cannot carry Pint units, the attribute ``mesh_units``
    lets users provide the unit which their mesh is defined in. Upon kernel dict
    creation the mesh is scaled to match the length unit used in the kernel.
    If ``mesh_units`` is not provided, no scaling of the mesh is performed.

    The :meth:`MeshTreeElement.from_dict` constructor instantiates the class from
    a configuration dictionary.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="mesh_tree_element",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="User-defined object identifier.",
        type="str, optional",
    )

    mesh_filename: Path = documented(
        attr.ib(
            converter=Path,
            kw_only=True,
        ),
        doc="Path to the triangulated mesh data file. This parameter is required.",
        type=":class:`pathlib.Path`",
        init_type="path-like",
    )

    @mesh_filename.validator
    def _mesh_filename_validator(self, attribute, value):
        validators.path_exists(self, attribute, value)

        if not value.suffix in [".obj", ".ply"]:
            raise ValueError(
                f"While validating {attribute.name}: File extension must be '.obj'"
                f"or '.ply', got {value.suffix}"
            )

    mesh_units: t.Optional[pint.Unit] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(ureg.Unit),
        ),
        doc="Units the mesh was defined in. Used to convert to kernel units. "
        "If this value is ``None``, the mesh is interpreted as being defined in"
        "kernel units.",
        type=":class:`pint.Unit`, optional",
        init_type="str or :class:`pint.Unit`, optional",
    )

    @mesh_units.validator
    def _mesh_units_validator(self, attribute, value):
        if not isinstance(value, pint.unit.Unit):
            raise ValueError(
                f"While validating {attribute.name}: Mesh unit parameter must be "
                f"a pint Unit object, got {type(value)}"
            )

    reflectance: Spectrum = documented(
        attr.ib(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance of the object. "
        "Must be a reflectance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        init_type=":class:`.Spectrum` or dict",
        default="0.5",
    )

    transmittance: Spectrum = documented(
        attr.ib(
            default=0.0,
            converter=spectrum_factory.converter("transmittance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("transmittance"),
            ],
        ),
        doc="Transmittance of the object. "
        "Must be a transmittance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        init_type=":class:`.Spectrum` or dict",
        default="0.0",
    )

    # --------------------------------------------------------------------------
    #                              Constructors
    # --------------------------------------------------------------------------

    @staticmethod
    def from_dict(d: t.Dict) -> MeshTreeElement:
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameters
        ----------
        d : dict
            Configuration dictionary used for initialisation.

        Returns
        -------
        :class:`.MeshTreeElement`
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return MeshTreeElement(**d_copy)

    @staticmethod
    def convert(value: t.Any) -> t.Any:
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`.from_dict` to
        create an :class:`.MeshTreeElement`. Otherwise, it returns ``value``.

        Parameters
        ----------
        value
            Value to attempt conversion of.

        Returns
        -------
        any type
            Converted value.
        """
        if isinstance(value, dict):
            return MeshTreeElement.from_dict(value)

        return value

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_bsdfs(self, ctx: KernelDictContext) -> t.Dict:
        return {
            f"bsdf_{self.id}": {
                "type": "bilambertian",
                "reflectance": self.reflectance.kernel_dict(ctx=ctx)["spectrum"],
                "transmittance": self.transmittance.kernel_dict(ctx=ctx)["spectrum"],
            }
        }

    def kernel_shapes(self, ctx: KernelDictContext) -> t.Dict:
        # Inherit docstring

        bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}

        if self.mesh_units is None:
            scaling_factor = 1.0
        else:
            kernel_length = uck.get("length")
            scaling_factor = (1.0 * self.mesh_units).m_as(kernel_length)

        base_dict = {
            "filename": str(self.mesh_filename),
            "bsdf": bsdf,
            "to_world": mi.ScalarTransform4f.scale(scaling_factor),
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
