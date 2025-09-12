from __future__ import annotations

import os
from abc import ABC, abstractmethod

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ..core import CompositeSceneElement
from ... import validators
from ..._factory import Factory
from ...attrs import define, documented, get_doc
from ...kernel import SceneParameter
from ...typing import PathLike
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import flatten

biosphere_factory = Factory()
biosphere_factory.register_lazy_batch(
    [
        (
            "_core.InstancedCanopyElement",
            "instanced",
            {},
        ),
        (
            "_discrete.DiscreteCanopy",
            "discrete_canopy",
            {"dict_constructor": "padded"},
        ),
        (
            "_leaf_cloud.LeafCloud",
            "leaf_cloud",
            {},
        ),
        (
            "_tree.AbstractTree",
            "abstract_tree",
            {},
        ),
        (
            "_tree.MeshTree",
            "mesh_tree",
            {},
        ),
    ],
    cls_prefix="eradiate.scenes.biosphere",
)


@define(eq=False, slots=False)
class Canopy(CompositeSceneElement, ABC):
    """
    Abstract base class for all canopies.
    """

    id: str | None = documented(
        attrs.field(
            default="canopy",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(CompositeSceneElement, "id", "doc"),
        type=get_doc(CompositeSceneElement, "id", "type"),
        init_type=get_doc(CompositeSceneElement, "id", "init_type"),
        default='"canopy"',
    )

    size: pint.Quantity | None = documented(
        pinttr.field(
            default=None,
            validator=attrs.validators.optional(
                [
                    pinttr.validators.has_compatible_units,
                    validators.on_quantity(validators.is_vector3),
                ]
            ),
            units=ucc.deferred("length"),
        ),
        doc="Canopy extent as a 3-vector.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
    )


@define(eq=False, slots=False)
class CanopyElement(CompositeSceneElement, ABC):
    """
    Abstract base class representing a component of a :class:`.Canopy` object.
    Concrete canopy classes can manage their components as they prefer.
    """

    @property
    @abstractmethod
    def _template_bsdfs(self) -> dict:
        pass

    @property
    @abstractmethod
    def _template_shapes(self) -> dict:
        pass

    @property
    def template(self) -> dict:
        return flatten({**self._template_bsdfs, **self._template_shapes})

    @property
    @abstractmethod
    def _params_bsdfs(self) -> dict:
        pass

    @property
    @abstractmethod
    def _params_shapes(self) -> dict:
        pass

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring
        return flatten({**self._params_bsdfs, **self._params_shapes})


@define(eq=False, slots=False)
class InstancedCanopyElement(CompositeSceneElement):
    """
    Instanced canopy element [``instanced``].

    This class wraps a canopy element and defines locations where to position
    instances (*i.e.* clones) of it.

    .. admonition:: Class method constructors

       .. autosummary::

          from_file
    """

    canopy_element: CanopyElement | None = documented(
        attrs.field(
            default=None,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(CanopyElement)
            ),
            converter=biosphere_factory.convert,
        ),
        doc="Instanced canopy element. Can be specified as a dictionary, which "
        "will be converted by :data:`.biosphere_factory`.",
        type=":class:`.CanopyElement`, optional",
    )

    instance_positions: pint.Quantity = documented(
        pinttr.field(
            factory=list,
            units=ucc.deferred("length"),
        ),
        doc="Instance positions as an (n, 3)-array.\n"
        "\n"
        "Unit-enabled field (default: ucc['length'])",
        type="quantity",
        init_type="array-like",
        default="[]",
    )

    @instance_positions.validator
    def _instance_positions_validator(self, attribute, value):
        if value.shape and value.shape[0] > 0 and value.shape[1] != 3:
            raise ValueError(
                f"while validating {attribute.name}, must be an array of shape "
                f"(n, 3), got {value.shape}"
            )

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        canopy_element: CanopyElement | None = None,
    ):
        """
        Construct a :class:`.InstancedCanopyElement` from a text file specifying
        instance positions.

        .. admonition:: File format

           Each line defines an instance position as a whitespace-separated
           3-vector of Cartesian coordinates.

        .. important::

           Location coordinates are assumed to be given in meters.

        Parameters
        ----------
        filename : path-like
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative.

        canopy_element : .CanopyElement or dict, optional
            :class:`.CanopyElement` to be instanced. If a dictionary is passed,
            if is interpreted by :data:`.biosphere_factory`. If set to
            ``None``, an empty leaf cloud will be created.

        Returns
        -------
        :class:`.InstancedCanopyElement`
            Created :class:`.InstancedCanopyElement`.

        Raises
        ------
        ValueError
            If ``filename`` is set to ``None``.

        FileNotFoundError
            If ``filename`` does not point to an existing file.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"no file at {filename} found.")

        if canopy_element is None:
            canopy_element = {"type": "leaf_cloud"}

        canopy_element = biosphere_factory.convert(canopy_element)

        instance_positions = []

        with open(filename, "r") as f:
            for i_line, line in enumerate(f):
                try:
                    coords = np.array(line.split(), dtype=float)
                except ValueError as e:
                    raise ValueError(
                        f"while reading {filename}, on line {i_line + 1}: "
                        f"cannot convert {line} to a 3-vector!"
                    ) from e

                if len(coords) != 3:
                    raise ValueError(
                        f"while reading {filename}, on line {i_line + 1}: "
                        f"cannot convert {line} to a 3-vector!"
                    )

                instance_positions.append(coords)

        instance_positions = np.array(instance_positions) * ureg.m
        return cls(canopy_element=canopy_element, instance_positions=instance_positions)

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _template_bsdfs(self) -> dict:
        # Produces the part of the kernel dictionary template describing
        # the BSDFs of the encapsulated canopy element
        return self.canopy_element._template_bsdfs

    @property
    def _template_shapes(self) -> dict:
        # Produces the part of the kernel dictionary template describing
        # the shape group of the encapsulated canopy element
        result = {f"{self.canopy_element.id}.type": "shapegroup"}

        for key, param in self.canopy_element._template_shapes.items():
            result[f"{self.canopy_element.id}.{key}"] = param

        return result

    @property
    def _template_instances(self) -> dict:
        # Produces the part of the kernel dictionary template describing
        # instances of the encapsulated canopy element
        length_units = uck.get("length")

        result = {}

        for i, position in enumerate(self.instance_positions.m_as(length_units)):
            result[f"{self.canopy_element.id}_instance_{i}.type"] = "instance"
            result[f"{self.canopy_element.id}_instance_{i}.group.type"] = "ref"
            result[f"{self.canopy_element.id}_instance_{i}.group.id"] = (
                self.canopy_element.id
            )
            result[f"{self.canopy_element.id}_instance_{i}.to_world"] = (
                mi.ScalarTransform4f.translate(position)
            )

        return result

    @property
    def template(self) -> dict:
        # Inherit docstring
        return {
            **self._template_bsdfs,
            **self._template_shapes,
            **self._template_instances,
        }

    @property
    def _params_bsdfs(self) -> dict:
        # Produces the part of the parameter map describing the BSDFs of the
        # encapsulated canopy element
        return self.canopy_element._params_bsdfs

    @property
    def _params_shapes(self) -> dict:
        # Produces the part of the parameter map describing the shape group of
        # the encapsulated canopy element
        return self.canopy_element._params_shapes

    @property
    def _params_instances(self) -> dict:
        # Produces the part of the parameter map describing instances of the
        # encapsulated canopy element
        return {}

    @property
    def params(self) -> dict:
        # Inherit docstring
        return {
            **self._params_bsdfs,
            **self._params_shapes,
            **self._params_instances,
        }
