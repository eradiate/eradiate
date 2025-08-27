from __future__ import annotations

import importlib
import typing as t
from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from .._factory import Factory
from ..attrs import define, documented, frozen
from ..exceptions import TraversalError
from ..kernel import KernelDict, KernelSceneParameterMap, SceneParameter
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

# ------------------------------------------------------------------------------
#                           Scene element interface
# ------------------------------------------------------------------------------


@define(eq=False, slots=False)
class SceneElement(ABC):
    """
    Abstract base class for all scene elements.

    Warnings
    --------
    All subclasses *must* have a hash, thus ``eq`` must be ``False`` (see
    `attrs docs on hashing <https://www.attrs.org/en/stable/hashing.html>`__
    for a complete explanation). This is required in order to make it possible
    to use caching decorators on instance methods.

    Notes
    -----
    The default implementation of ``__attrs_post_init__()`` executes the
    :meth:`update` method.
    """

    id: str | None = documented(
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
    def params(self) -> dict[str, SceneParameter] | None:
        """
        Returns
        -------
        dict[str, :class:`.SceneParameter`] or None
            A dictionary mapping parameter paths, consisting of dot-separated
            strings, to a corresponding update protocol.

        See Also
        --------
        :class:`.SceneParameter`, :class:`.KernelSceneParameterMap`
        """
        return None

    @abstractmethod
    def traverse(self, callback) -> None:
        """
        Traverse this scene element and collect kernel dictionary template and
        parameter update map contributions.

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


@define(eq=False, slots=False)
class NodeSceneElement(SceneElement, ABC):
    """
    Abstract base class for scene elements which expand as a single Mitsuba
    scene tree node which can be described as a scene dictionary.
    """

    @property
    @abstractmethod
    def template(self) -> dict:
        """
        Kernel dictionary template contents associated with this scene element.

        Returns
        -------
        dict
            A flat dictionary mapping dot-separated strings describing the path
            of an item in the nested scene dictionary to values. Values may be
            objects which can be directly used by the :func:`mitsuba.load_dict`
            function, or :class:`.DictParameter` instances which must be
            rendered.

        See Also
        --------
        :class:`.DictParameter`, :class:`.KernelDict`
        """
        pass

    @property
    def objects(self) -> dict[str, NodeSceneElement] | None:
        """
        Map of child objects associated with this scene element.

        Returns
        -------
        dict
            A dictionary mapping object names to a corresponding object to be
            inserted in the Eradiate scene graph.
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


@define(eq=False, slots=False)
class InstanceSceneElement(SceneElement, ABC):
    """
    Abstract base class for scene elements which represent a node in the Mitsuba
    scene graph, but can only be expanded to a Mitsuba object.
    """

    @property
    @abstractmethod
    def instance(self) -> mi.Object:
        """
        Mitsuba object which is represented by this scene element.

        Returns
        -------
        mitsuba.Object
        """

        pass

    def traverse(self, callback):
        # Inherit docstring
        callback.put_instance(self.instance)

        if self.params is not None:
            callback.put_params(self.params)


@define(eq=False, slots=False)
class CompositeSceneElement(SceneElement, ABC):
    """
    Abstract based class for scene elements which expand to multiple Mitsuba
    scene tree nodes.
    """

    @property
    def template(self) -> dict:
        """
        Kernel dictionary template contents associated with this scene element.

        Returns
        -------
        dict
            A flat dictionary mapping dot-separated strings describing the path
            of an item in the nested scene dictionary to values. Values may be
            objects which can be directly used by the :func:`mitsuba.load_dict`
            function, or :class:`.DictParameter` instances which must be
            rendered.

        See Also
        --------
        :class:`.DictParameter`, :class:`.KernelDict`
        """
        return {}

    @property
    def objects(self) -> dict[str, NodeSceneElement] | None:
        """
        Map of child objects associated with this scene element.

        Returns
        -------
        dict
            A dictionary mapping object names to a corresponding object to be
            inserted in the Eradiate scene graph.
        """
        return None

    def traverse(self, callback):
        # Inherit docstring
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


@define(eq=False, slots=False)
class Ref(NodeSceneElement):
    """
    A scene element which represents a reference to a Mitsuba scene tree node.
    """

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
        # Inherit docstring
        return {"type": "ref", "id": self.id}


@define(eq=False, slots=False)
class Scene(NodeSceneElement):
    """
    A generic scene element container which expands as a :class:`mitsuba.Scene`
    object.
    """

    _objects: dict[str, SceneElement] = documented(
        attrs.field(factory=dict, converter=dict),
        doc="A map of scene elements which will be included in the Mitsuba "
        "scene definition.",
        type="dict",
        default="{}",
    )

    @property
    def template(self) -> dict:
        # Inherit docstring
        return {"type": "scene"}

    @property
    def objects(self) -> dict[str, SceneElement]:
        # Inherit docstring
        return self._objects


# ------------------------------------------------------------------------------
#                         Scene element tree traversal
# ------------------------------------------------------------------------------


@define
class SceneTraversal:
    """
    Data structure used to collect kernel dictionary data during scene element
    traversal.
    """

    #: Current traversal node
    node: NodeSceneElement

    #: Parent to current node
    parent: NodeSceneElement | None = attrs.field(default=None)

    #: Current node's name
    name: str | None = attrs.field(default=None)

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

    def put_template(self, template: t.Mapping) -> None:
        """
        Add a contribution to the kernel dictionary template.
        """
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in template.items():
            self.template[f"{prefix}{k}"] = v

    def put_params(self, params: t.Mapping) -> None:
        """
        Add a contribution to the parameter map.
        """
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in params.items():
            self.params[f"{prefix}{k}"] = v

    def put_object(self, name: str, node: SceneElement) -> None:
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

    def put_instance(self, obj: mi.Object) -> None:
        """
        Add an instance to the kernel dictionary template.
        """
        if not self.name:
            raise TraversalError("Instances may only be inserted as child nodes.")
        self.template[self.name] = obj


def traverse(node: NodeSceneElement) -> tuple[KernelDict, KernelSceneParameterMap]:
    """
    Traverse a scene element tree and collect kernel dictionary template and
    parameter update table data.

    Parameters
    ----------
    node : .SceneElement
        Scene element where to start traversal.

    Returns
    -------
    kdict_template : .KernelDict
        Kernel dictionary template corresponding to the traversed scene element.

    umap_template : .KernelSceneParameterMap
        Kernel parameter table associated with the traversed scene element.
    """
    # Traverse scene element tree
    cb = SceneTraversal(node)
    node.traverse(cb)

    # Use collected data to generate the kernel dictionary
    return KernelDict(cb.template), KernelSceneParameterMap(cb.params)


# -- Misc (to be moved elsewhere) ----------------------------------------------


@frozen
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
        if not np.all(np.less_equal(self.min, self.max)):
            raise ValueError(
                f"while validating {attribute.name}: 'min' must be less or "
                "equal to 'max'"
            )

    @classmethod
    def convert(
        cls, value: t.Sequence | t.Mapping | np.typing.ArrayLike | pint.Quantity
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
            ``ucc['length']``.

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
