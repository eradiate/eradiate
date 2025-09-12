from __future__ import annotations

import typing as t
from abc import ABC
from collections.abc import MutableMapping
from pathlib import Path

import attrs
import mitsuba as mi
import pint
import pinttr

from ._core import CanopyElement, biosphere_factory
from ._leaf_cloud import LeafCloud
from ..core import SceneElement, traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented, get_doc
from ...kernel import SceneParameter, SearchSceneParameter
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@define(eq=False, slots=False)
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


@define(eq=False, slots=False)
class AbstractTree(Tree):
    """
    A container class for abstract trees in discrete canopies.
    Holds a :class:`.LeafCloud` and the parameters characterizing a cylindrical
    trunk. The entire tree is described in local coordinates and can be placed
    in the scene using :class:`.InstancedCanopyElement`.

    The trunk starts at [0, 0, -0.1] and extends
    to [0, 0, trunk_height]. The trunk extends below ``z=0`` to avoid
    intersection issues at the intersection of the trunk and the ground the tree
    is usually placed on.

    The leaf cloud will by default be offset such that its local coordinate
    origin coincides with the upper end of the trunk. If this is not desired,
    e.g. the leaf cloud is centered around its coordinate origin and the trunk
    should not extend into it, the parameter ``leaf_cloud_extra_offset`` can be
    used to shift the leaf cloud **in addition** to the trunk's extent.
    """

    id: str | None = documented(
        attrs.field(
            default="abstract_tree",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"abstract_tree"',
    )

    leaf_cloud: LeafCloud | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(_leaf_cloud_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(LeafCloud)
            ),
        ),
        doc="Instanced leaf cloud. Can be specified as a dictionary, which will "
        "be interpreted by :data:`.biosphere_factory`. If the latter case, the "
        '``"type"`` parameter, if omitted, will implicitly be set to '
        '``"leaf_cloud"``.',
        type=".LeafCloud or None",
        init_type=".LeafCloud or dict, optional",
    )

    trunk_height: pint.Quantity = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Trunk height. Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float",
        default="1.0 m",
    )

    trunk_radius: pint.Quantity = documented(
        pinttr.field(default=0.1 * ureg.m, units=ucc.deferred("length")),
        doc="Trunk radius. Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float, optional",
        default="0.1 m",
    )

    trunk_reflectance: Spectrum = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
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
        pinttr.field(factory=lambda: [0, 0, 0], units=ucc.deferred("length")),
        doc="Additional offset for the leaf cloud. 3-vector. "
        "Unit-enabled field (default: ucc['length'])",
        type="quantity",
        init_type="quantity or array-like, optional",
        default="[0, 0, 0]",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def bsdf_id(self) -> str:
        return f"bsdf_{self.id}"

    @property
    def _template_bsdfs(self) -> dict:
        objects = {"reflectance": traverse(self.trunk_reflectance)[0].data}

        result = {f"{self.bsdf_id}.type": "diffuse"}

        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{self.bsdf_id}.{obj_key}.{key}"] = param

        return {**self.leaf_cloud._template_bsdfs, **result}

    @property
    def _template_shapes(self) -> dict:
        kernel_length = uck.get("length")

        kernel_height = self.trunk_height.m_as(kernel_length)
        kernel_radius = self.trunk_radius.m_as(kernel_length)

        leaf_cloud = self.leaf_cloud.translated(
            [0.0, 0.0, kernel_height] * kernel_length
            + self.leaf_cloud_extra_offset.to(kernel_length)
        )

        bsdf = {"type": "ref", "id": self.bsdf_id}

        result = {
            **leaf_cloud._template_shapes,
            f"trunk_cyl_{self.id}": {
                "type": "cylinder",
                "bsdf": bsdf,
                "radius": kernel_radius,
                "p0": [0, 0, -0.1],
                "p1": [0, 0, kernel_height],
            },
            f"trunk_cap_{self.id}": {
                "type": "disk",
                "bsdf": bsdf,
                "to_world": mi.ScalarTransform4f.scale(kernel_radius)
                @ mi.ScalarTransform4f.translate([0, 0, kernel_height]),
            },
        }

        return result

    @property
    def _params_bsdfs(self) -> dict:
        objects = {"reflectance": traverse(self.trunk_reflectance)[1].data}

        result = {}

        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                # If no lookup strategy is set, we must add one
                if isinstance(param, SceneParameter) and param.search is None:
                    param = attrs.evolve(
                        param,
                        search=SearchSceneParameter(
                            mi.BSDF,
                            self.bsdf_id,
                            parameter_relpath=f"{obj_key}.{key}",
                        ),
                    )

                result[f"{self.bsdf_id}.{obj_key}.{key}"] = param

        for key, param in self.leaf_cloud._params_bsdfs.items():
            result[key] = param

        return result

    @property
    def _params_shapes(self) -> dict:
        return {}


@define(eq=False, slots=False)
class MeshTree(Tree):
    """
    A container class for mesh based tree-like objects in canopies.

    It holds one or more triangulated meshes and corresponding BSDFs, representing
    the tree.

    The mesh will be interpreted in local coordinates and should be used in an
    :class:`.InstancedCanopyElement` to place at arbitrary positions in a scene.
    """

    id: str = documented(
        attrs.field(
            default="mesh_tree",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"mesh_tree"',
    )

    mesh_tree_elements: list[MeshTree] = documented(
        attrs.field(
            factory=list,
            converter=lambda value: [
                MeshTreeElement.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [MeshTreeElement.convert(value)],
        ),
        doc="List of :class:`.CanopyElement` defining the canopy. Can be "
        "initialized with a :class:`.InstancedCanopyElement`, which will be "
        "automatically wrapped into a list. Dictionary-based specifications are "
        "allowed as well.",
        type="list of .InstancedCanopyElement",
        init_type="list of (.InstancedCanopyElement | dict)",
        default="[]",
    )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _template_bsdfs(self) -> dict:
        result = {}
        for element in self.mesh_tree_elements:
            result.update(element._template_bsdfs)
        return result

    @property
    def _template_shapes(self) -> dict:
        result = {}
        for element in self.mesh_tree_elements:
            result.update(element._template_shapes)
        return result

    @property
    def _params_bsdfs(self) -> dict:
        result = {}
        for element in self.mesh_tree_elements:
            result.update(element._params_bsdfs)
        return result

    @property
    def _params_shapes(self) -> dict:
        return {}


@define(eq=False, slots=False)
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

    id: str | None = documented(
        attrs.field(
            default="mesh_tree_element",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="User-defined object identifier.",
        type="str, optional",
    )

    mesh_filename: Path = documented(
        attrs.field(
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

        if value.suffix not in {".obj", ".ply"}:
            raise ValueError(
                f"While validating {attribute.name}: File extension must be '.obj'"
                f"or '.ply', got '{value.suffix}'"
            )

    mesh_units: pint.Unit | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(ureg.Unit),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(pint.Unit)
            ),
        ),
        doc="Units the mesh was defined in. Used to convert to kernel units. "
        "If unset, the mesh is interpreted as being defined in kernel units.",
        type=":class:`pint.Unit`, optional",
        init_type="str or :class:`pint.Unit`, optional",
    )

    reflectance: Spectrum = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
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
        attrs.field(
            default=0.0,
            converter=spectrum_factory.converter("transmittance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
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
    def from_dict(d: dict) -> MeshTreeElement:
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameters
        ----------
        d : dict
            Configuration dictionary used for initialization.

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

    @property
    def bsdf_id(self) -> str:
        return f"bsdf_{self.id}"

    @property
    def _template_bsdfs(self) -> dict:
        objects = {
            "reflectance": traverse(self.reflectance)[0].data,
            "transmittance": traverse(self.transmittance)[0].data,
        }

        result = {f"{self.bsdf_id}.type": "bilambertian"}

        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                result[f"{self.bsdf_id}.{obj_key}.{key}"] = param

        return result

    @property
    def _template_shapes(self) -> dict:
        scaling_factor = (
            1.0
            if self.mesh_units is None
            else ureg.convert(1.0, self.mesh_units, uck.get("length"))
        )

        if self.mesh_filename.suffix == ".obj":
            shape_type = "obj"
        elif self.mesh_filename.suffix == ".ply":
            shape_type = "ply"
        else:
            raise ValueError(
                f"unsupported file extension '{self.mesh_filename.suffix}'"
            )

        result = {
            f"{self.id}.type": shape_type,
            f"{self.id}.bsdf.type": "ref",
            f"{self.id}.bsdf.id": self.bsdf_id,
            f"{self.id}.filename": str(self.mesh_filename),
            f"{self.id}.to_world": mi.ScalarTransform4f.scale(scaling_factor),
        }

        return result

    @property
    def _params_bsdfs(self) -> dict:
        objects = {
            "reflectance": traverse(self.reflectance)[1].data,
            "transmittance": traverse(self.transmittance)[1].data,
        }

        result = {}

        for obj_key, obj_params in objects.items():
            for key, param in obj_params.items():
                # If no lookup strategy is set, we must add one
                if isinstance(param, SceneParameter) and param.search is None:
                    param = attrs.evolve(
                        param,
                        search=SearchSceneParameter(
                            mi.BSDF,
                            self.bsdf_id,
                            parameter_relpath=f"{obj_key}.{key}",
                        ),
                    )

                result[f"{self.bsdf_id}.{obj_key}.{key}"] = param

        return result

    @property
    def _params_shapes(self) -> dict:
        return {}
