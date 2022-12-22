from __future__ import annotations

import enum
import re
import typing as t
from collections import UserDict
from typing import Mapping, Sequence

import attrs
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from ..attrs import documented, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import TraversalError
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import nest

# ------------------------------------------------------------------------------
#                             Kernel dictionary
# ------------------------------------------------------------------------------


class ParamFlags(enum.Flag):
    """
    Parameter flags.
    """

    NONE = 0
    INIT = enum.auto()  #: Used for kernel dict generation
    SPECTRAL = enum.auto()  #: Varies during the spectral loop
    GEOMETRIC = enum.auto()  #: Triggers a scene rebuild
    ALL = INIT | SPECTRAL | GEOMETRIC
    UPDATE = SPECTRAL | GEOMETRIC  #: Update-specific flags


class _Unused:
    """
    Sentinel class to indicate when a parameter is unused, accessible as
    :data:`.Param.UNUSED`.

    Notes
    -----
    ``bool(_Unused)`` evaluates to ``False``.
    """

    _singleton = None

    def __new__(cls):
        if _Unused._singleton is None:
            _Unused._singleton = super(_Unused, cls).__new__(cls)
        return _Unused._singleton

    def __repr__(self):
        return "UNUSED"

    def __bool__(self):
        return False

    def __len__(self):
        return 0


@attrs.define
class Param:
    """
    A kernel scene parameter generator.
    """

    #: Sentinel value indicated that a parameter is not used
    UNUSED: t.ClassVar = _Unused()

    #: An attached callable which evaluates the parameter.
    _callable: t.Callable = attrs.field(repr=False)

    #: Flags specifying parameter attributes. By default, the declared parameter
    #: will pass all filters.
    flags: ParamFlags = attrs.field(default=ParamFlags.ALL)

    def __call__(self, ctx: KernelDictContext) -> t.Any:
        return self._callable(ctx)


@attrs.define(slots=False)
class ParameterMap(UserDict):
    """
    A dict-like structure mapping parameter paths to methods generating them.
    """

    data: dict[str, Param] = attrs.field(factory=dict)

    def remove(self, keys: t.Union[str, t.List[str]]) -> None:
        """
        Remove all parameters matching the given regular expression.

        Parameters
        ----------
        keys : str or list of str
            Regular expressions matching the parameters to remove.

        Notes
        -----
        This method mutates the parameter map.
        """
        if not isinstance(keys, list):
            keys = [keys]

        import re

        regexps = [re.compile(k).match for k in keys]
        keys = [k for k in self.keys() if any(r(k) for r in regexps)]

        for key in keys:
            del self.data[key]

    def keep(self, keys: t.Union[str, t.List[str]]) -> None:
        """
        Keep only parameters matching the given regular expression.

        Parameters
        ----------
        keys : str or list of str
            Regular expressions matching the parameters to keep.

        Notes
        -----
        This method mutates the parameter map.
        """
        if not isinstance(keys, list):
            keys = [keys]

        import re

        regexps = [re.compile(k).match for k in keys]
        keys = [k for k in self.keys() if any(r(k) for r in regexps)]
        result = {k: self.data[k] for k in keys}
        self.data = result

    def render(
        self,
        ctx: KernelDictContext,
        flags: ParamFlags = ParamFlags.ALL,
        drop: bool = False,
    ) -> dict:
        """
        Evaluate the parameter map for a set of arguments.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A kernel dictionary context.

        flags : :class:`.ParamFlags`
            Parameter flags. Only parameters with at least one of the specified
            will pass the filter.

        drop : bool
            If ``True``, drop unused parameters. Parameters may be unused either
            because they were filtered out by the flags or because context
            information implied it.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If ``drop`` is ``False`` and the rendered parameter map contains an
            unused parameter.
        """
        result = self.data.copy()
        unused = render_params(result, ctx=ctx, flags=flags, drop=drop)

        # Check for leftover empty values
        if not drop and unused:
            raise ValueError(f"Unevaluated parameters: {unused}")

        return result


@attrs.define
class KernelDictTemplate(ParameterMap):
    """
    A dict-like structure which contains the structure of an instantiable kernel
    dictionary.

    Entries are indexed by dot-separated paths which can then be expanded to
    a nested dictionary using the :meth:`.render` method.

    Each entry can be either a hard-coded value which can be directly
    interpreted by the :func:`mitsuba.load_dict` function, or a :class:`.Param`
    object, which must be rendered before the template can be instantiated.
    """

    def render(
        self,
        ctx: KernelDictContext,
        flags: ParamFlags = ParamFlags.INIT,
        drop: bool = False,
        nested: bool = True,
    ) -> dict:
        """
        Render the template as a nested dictionary using a parameter map to fill
        in empty fields.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A kernel dictionary context.

        flags : :class:`.ParamFlags`
            Parameter flags. Only parameters with at least one of the specified
            will pass the filter.

        drop : bool
            If ``True``, drop unused parameters. Parameters may be unused either
            because they were filtered out by the flags or because context
            information implied it.

        nested : bool, optional
            If ``True``, the returned dictionary will be nested and suitable for
            instantiation by Mitsuba; otherwise, the returned dictionary will be
            flat.

        Returns
        -------
        dict
        """
        result = super().render(ctx=ctx, flags=flags, drop=drop)
        return nest(result, sep=".") if nested else result


def render_params(
    d: t.MutableMapping,
    ctx: KernelDictContext,
    flags: ParamFlags = ParamFlags.ALL,
    drop: bool = False,
) -> list:
    """
    Render parameters in a template dictionary.

    Parameters
    ----------
    d : dict-like
        A dict-like containing parameters to render. *In-place* modification
        will be performed.

    ctx : :class:`.KernelDictContext`
        A kernel dictionary context.

    flags : :class:`.ParamFlags`
        Parameter flags. Only parameters with at least one of the specified
        will pass the filter.

    drop : bool
        If ``True``, drop unused parameters.

    Returns
    -------
    list
        A list of unused parameters. A parameter may be unused because it was
        filtered out by flags or because context information implied it.
    """
    unused = []

    for k, v in d.items():
        if isinstance(v, Param):
            if v.flags & flags:
                d[k] = v(ctx)
            else:
                d[k] = Param.UNUSED

            if d[k] is Param.UNUSED:
                unused.append(k)

    if drop:
        for k in unused:
            del d[k]

    return unused


# ------------------------------------------------------------------------------
#                           Scene element interface
# ------------------------------------------------------------------------------


@attrs.define(eq=False, slots=False)
class SceneElement:
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
    def params(self) -> t.Optional[t.Dict[str, Param]]:
        """
        Map of updatable parameters associated with this scene element.
        """
        return None

    def traverse(self, callback):
        """
        Traverse this scene element and collect kernel dictionary template,
        parameter and object map contributions.

        Parameters
        ----------
        callback : SceneTraversal
            Callback data structure storing the collected data.
        """
        raise NotImplementedError

    def update(self) -> None:
        """
        Enforce internal state consistency. This method should be called when
        fields are modified. It is automatically called as a post-init step.
        """
        # The default implementation is a no-op
        pass


@attrs.define(eq=False, slots=False)
class NodeSceneElement(SceneElement):
    @property
    def template(self) -> dict:
        """
        Kernel dictionary template contents associated with this scene element.
        """
        raise NotImplementedError

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
class InstanceSceneElement(SceneElement):
    @property
    def instance(self) -> "mitsuba.Object":
        raise NotImplementedError

    def traverse(self, callback):
        callback.put_instance(self.instance)

        if self.params is not None:
            callback.put_params(self.params)


@attrs.define(eq=False, slots=False)
class CompositeSceneElement(SceneElement):
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
    _objects: t.Dict[str, NodeSceneElement] = attrs.field(factory=dict, converter=dict)

    @property
    def template(self) -> dict:
        return {"type": "scene"}

    @property
    def objects(self) -> t.Dict[str, NodeSceneElement]:
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


def traverse(node: NodeSceneElement) -> t.Tuple[KernelDictTemplate, ParameterMap]:
    """
    Traverse a scene element tree and collect kernel dictionary data.

    Parameters
    ----------
    node : .SceneElement
        Scene element where to start traversal.

    Returns
    -------
    template : .KernelDictTemplate
        Kernel dictionary template corresponding to the traversed scene element.

    params : .ParameterMap
        Kernel parameter table associated with the traversed scene element.
    """
    # Traverse scene element tree
    cb = SceneTraversal(node)
    node.traverse(cb)

    # Use collected data to generate the kernel dictionary
    return KernelDictTemplate(cb.template), ParameterMap(cb.params)


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
