from __future__ import annotations

import importlib
import typing as t
from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import attrs
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from .._factory import Factory
from ..attrs import documented, parse_docs
from ..exceptions import TraversalError
from ..kernel import (
    KernelDictTemplate,
    UpdateMapTemplate,
    UpdateParameter,
)
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

# ------------------------------------------------------------------------------
#                           Scene element interface
# ------------------------------------------------------------------------------


@attrs.define(eq=False, slots=False)
class SceneElement(ABC):
    """
    Important: All subclasses *must* have a hash, thus eq must be False (see
    attrs docs on hashing for a complete explanation).
    """

    id: t.Optional[str] = documented(
        attrs.field(
            default=None,
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="Identifier of the current scene element.",
        type="str or None",
        init_type="str, optional",
    )

    def __attrs_post_init__(self):
        self.update()

    @property
    def params(self) -> t.Optional[t.Dict[str, UpdateParameter]]:
        """
        Map of updatable parameters associated with this scene element.
        """
        return None

    @abstractmethod
    def traverse(self, callback):
        """
        Traverse this scene element and collect kernel dictionary template,
        parameter and object map contributions.

        Parameters
        ----------
        callback : SceneTraversal
            Callback data structure storing the collected data.
        """
        pass

    def update(self) -> None:
        """
        Enforce internal state consistency. This method should be called when
        fields are modified. It is automatically called as a post-init step.
        """
        # The default implementation is a no-op
        pass


@attrs.define(eq=False, slots=False)
class NodeSceneElement(SceneElement, ABC):
    @property
    @abstractmethod
    def template(self) -> dict:
        """
        Kernel dictionary template contents associated with this scene element.
        """
        pass

    @property
    def objects(self) -> t.Optional[t.Dict[str, NodeSceneElement]]:
        """
        Map of child objects associated with this scene element.
        """
        return None

    def traverse(self, callback: SceneTraversal) -> None:
        # Inherit docstring
        callback.put_template(self.template)

        if self.params is not None:
            callback.put_params(self.params)

        if self.objects is not None:
            for name, obj in self.objects.items():
                callback.put_object(name, obj)


@attrs.define(eq=False, slots=False)
class InstanceSceneElement(SceneElement, ABC):
    @property
    @abstractmethod
    def instance(self) -> "mitsuba.Object":
        pass

    def traverse(self, callback):
        callback.put_instance(self.instance)

        if self.params is not None:
            callback.put_params(self.params)


@attrs.define(eq=False, slots=False)
class CompositeSceneElement(SceneElement, ABC):
    @property
    def template(self) -> dict:
        # The default implementation returns an empty dictionary
        return {}

    @property
    def objects(self) -> t.Optional[t.Dict[str, NodeSceneElement]]:
        """
        Map of child objects associated with this scene element.
        """
        return None

    def traverse(self, callback):
        callback.put_template(self.template)

        if self.params is not None:
            callback.put_params(self.params)

        if self.objects is not None:
            for name, obj in self.objects.items():
                if isinstance(obj, InstanceSceneElement):
                    callback.put_object(name, obj)

                else:
                    template, params = traverse(obj)
                    callback.put_template(
                        {f"{name}.{k}": v for k, v in template.items()}
                    )
                    callback.put_params({f"{name}.{k}": v for k, v in params.items()})


@attrs.define(eq=False, slots=False)
class Ref(NodeSceneElement):
    id: str = documented(
        attrs.field(
            kw_only=True,
            validator=attrs.validators.instance_of(str),
        ),
        doc="Identifier of the referenced kernel scene object (required).",
        type="str",
    )

    @property
    def template(self) -> dict:
        return {"type": "ref", "id": self.id}


@attrs.define(eq=False, slots=False)
class Scene(NodeSceneElement):
    _objects: t.Dict[str, SceneElement] = attrs.field(factory=dict, converter=dict)

    @property
    def template(self) -> dict:
        return {"type": "scene"}

    @property
    def objects(self) -> t.Dict[str, SceneElement]:
        return self._objects


# ------------------------------------------------------------------------------
#                         Scene element tree traversal
# ------------------------------------------------------------------------------


@attrs.define
class SceneTraversal:
    """
    Data structure used to collect kernel dictionary data during scene element
    traversal.
    """

    #: Current traversal node
    node: NodeSceneElement

    #: Parent to current node
    parent: t.Optional[NodeSceneElement] = attrs.field(default=None)

    #: Current node's name
    name: t.Optional[str] = attrs.field(default=None)

    #: Current depth
    depth: int = attrs.field(default=0)

    #: Dictionary mapping nodes to their parents
    hierarchy: dict = attrs.field(factory=dict)

    #: Kernel dictionary template
    template: dict = attrs.field(factory=dict)

    #: Dictionary mapping nodes to their defined parameters
    params: dict = attrs.field(factory=dict)

    def __attrs_post_init__(self):
        self.hierarchy[self.node] = (self.parent, self.depth)

    def put_template(self, template: t.Mapping):
        """
        Add a contribution to the kernel dictionary template.
        """
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in template.items():
            self.template[f"{prefix}{k}"] = v

    def put_params(self, params: t.Mapping):
        """
        Add a contribution to the parameter map.
        """
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in params.items():
            self.params[f"{prefix}{k}"] = v

    def put_object(self, name: str, node: SceneElement):
        """
        Add a child object to the template and parameter map.
        """
        if isinstance(node, CompositeSceneElement):
            node.traverse(self)

        else:
            if node is None or node in self.hierarchy:
                return

            cb = type(self)(
                node=node,
                parent=self.node,
                name=name if self.name is None else f"{self.name}.{name}",
                depth=self.depth + 1,
                hierarchy=self.hierarchy,
                template=self.template,
                params=self.params,
            )

            if isinstance(node, InstanceSceneElement):
                cb.put_instance(node.instance)
                cb.put_params(node.params)

            else:
                node.traverse(cb)

    def put_instance(self, obj: "mitsuba.Object"):
        """
        Add an instance to the kernel dictionary template.
        """
        if not self.name:
            raise TraversalError("Instances may only be inserted as child nodes.")
        self.template[self.name] = obj


def traverse(node: NodeSceneElement) -> t.Tuple[KernelDictTemplate, UpdateMapTemplate]:
    """
    Traverse a scene element tree and collect kernel dictionary data.

    Parameters
    ----------
    node : .SceneElement
        Scene element where to start traversal.

    Returns
    -------
    kdict_template : .KernelDictTemplate
        Kernel dictionary template corresponding to the traversed scene element.

    umap_template : .UpdateMapTemplate
        Kernel parameter table associated with the traversed scene element.
    """
    # Traverse scene element tree
    cb = SceneTraversal(node)
    node.traverse(cb)

    # Use collected data to generate the kernel dictionary
    return KernelDictTemplate(cb.template), UpdateMapTemplate(cb.params)


# -- Misc (to be moved elsewhere) ----------------------------------------------


@parse_docs
@attrs.frozen
class BoundingBox:
    """
    A basic data class representing an axis-aligned bounding box with
    unit-valued corners.

    Notes
    -----
    Instances are immutable.
    """

    min: pint.Quantity = documented(
        pinttr.field(
            units=ucc.get("length"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        type="quantity",
        init_type="array-like or quantity",
        doc="Min corner.",
    )

    max: pint.Quantity = documented(
        pinttr.field(
            units=ucc.get("length"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        type="quantity",
        init_type="array-like or quantity",
        doc="Max corner.",
    )

    @min.validator
    @max.validator
    def _min_max_validator(self, attribute, value):
        if not self.min.shape == self.max.shape:
            raise ValueError(
                f"while validating {attribute.name}: 'min' and 'max' must "
                f"have the same shape (got {self.min.shape} and {self.max.shape})"
            )
        if not np.all(np.less(self.min, self.max)):
            raise ValueError(
                f"while validating {attribute.name}: 'min' must be strictly "
                "less than 'max'"
            )

    @classmethod
    def convert(
        cls, value: t.Union[t.Sequence, t.Mapping, np.typing.ArrayLike, pint.Quantity]
    ) -> t.Any:
        """
        Attempt conversion of a value to a :class:`BoundingBox`.

        Parameters
        ----------
        value
            Value to convert.

        Returns
        -------
        any
            If `value` is an array-like, a quantity or a mapping, conversion will
            be attempted. Otherwise, `value` is returned unmodified.
        """
        if isinstance(value, (np.ndarray, pint.Quantity)):
            return cls(value[0, :], value[1, :])

        elif isinstance(value, Sequence):
            return cls(*value)

        elif isinstance(value, Mapping):
            return cls(**pinttr.interpret_units(value, ureg=ureg))

        else:
            return value

    @property
    def shape(self):
        """
        tuple: Shape of `min` and `max` arrays.
        """
        return self.min.shape

    @property
    def extents(self) -> pint.Quantity:
        """
        :class:`pint.Quantity`: Extent in all dimensions.
        """
        return self.max - self.min

    @property
    def units(self):
        """
        :class:`pint.Unit`: Units of `min` and `max` arrays.
        """
        return self.min.units

    def contains(self, p: np.typing.ArrayLike, strict: bool = False) -> bool:
        """
        Test whether a point lies within the bounding box.

        Parameters
        ----------
        p : quantity or array-like
            An array of shape (3,) (resp. (N, 3)) representing one (resp. N)
            points. If a unitless value is passed, it is interpreted as
            ``ucc["length"]``.

        strict : bool
            If ``True``, comparison is done using strict inequalities (<, >).

        Returns
        -------
        result : array of bool or bool
            ``True`` iff ``p`` in within the bounding box.
        """
        p = np.atleast_2d(ensure_units(p, ucc.get("length")))

        cmp = (
            np.logical_and(p > self.min, p < self.max)
            if strict
            else np.logical_and(p >= self.min, p <= self.max)
        )

        return np.all(cmp, axis=1)


# ------------------------------------------------------------------------------
#                               Factory accessor
# ------------------------------------------------------------------------------

_FACTORIES = {
    "atmosphere": "atmosphere.atmosphere_factory",
    "biosphere": "biosphere.biosphere_factory",
    "bsdf": "bsdfs.bsdf_factory",
    "illumination": "illumination.illumination_factory",
    "integrator": "integrators.integrator_factory",
    "measure": "measure.measure_factory",
    "phase": "phase.phase_function_factory",
    "shape": "shapes.shape_factory",
    "spectrum": "spectra.spectrum_factory",
    "surface": "surface.surface_factory",
}


def get_factory(element_type: str) -> Factory:
    """
    Return the factory corresponding to a scene element type.

    Parameters
    ----------
    element_type : str
        String identity of the scene element type associated to the requested
        factory.

    Returns
    -------
    factory : Factory
        Factory corresponding to the requested scene element type.

    Raises
    ------
    ValueError
        If the requested scene element type is unknown.

    Notes
    -----
    The ``element_type`` argument value maps to factories as follows:

    .. list-table::
       :widths: 1 1
       :header-rows: 1

       * - Element type ID
         - Factory
       * ``"atmosphere"``
         - :attr:`atmosphere_factory`
       * ``"biosphere"``
         - :attr:`biosphere_factory`
       * ``"bsdf"``
         - :attr:`bsdf_factory`
       * ``"illumination"``
         - :attr:`illumination_factory`
       * ``"integrator"``
         - :attr:`integrator_factory`
       * ``"measure"``
         - :attr:`measure_factory`
       * ``"phase"``
         - :attr:`phase_function_factory`
       * ``"shape"``
         - :attr:`shape_factory`
       * ``"spectrum"``
         - :attr:`spectrum_factory`
       * ``"surface"``
         - :attr:`surface_factory`
    """
    try:
        path = f"eradiate.scenes.{_FACTORIES[element_type]}"
    except KeyError:
        raise ValueError(
            f"unknown scene element type '{element_type}' "
            f"(should be one of {set(_FACTORIES.keys())})"
        )

    mod_path, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), attr)
