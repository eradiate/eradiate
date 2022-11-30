from __future__ import annotations

import enum
import typing as t
from collections import UserDict
from typing import Mapping, Sequence

import attrs
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from ..attrs import documented, parse_docs
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
    SPECTRAL = enum.auto()  #: Varies during the spectral loop
    GEOMETRIC = enum.auto()  #: Triggers a scene rebuild
    ALL = SPECTRAL | GEOMETRIC


@attrs.define
class Param:
    """
    A kernel scene parameter generator.
    """

    #: An attached callable which evaluates the parameter.
    _callable: t.Callable = attrs.field(repr=False)

    #: Flags specifying parameter attributes. By default, the declared parameter
    #: will pass all filters.
    flags: ParamFlags = attrs.field(default=ParamFlags.ALL)

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)


@attrs.define
class ParameterMap(UserDict):
    """
    A dict-like structure mapping parameter paths to methods generating them.
    """

    data: dict[str, Param] = attrs.field(factory=dict)

    def render(
        self, ctx, flags: ParamFlags = ParamFlags.ALL, drop: bool = False
    ) -> dict:
        """
        Evaluate the parameter map for a set of arguments.
        """
        result = self.data.copy()
        render_params(result, ctx=ctx, flags=flags, drop=drop)
        return result


@attrs.define
class KernelDictTemplate(UserDict):
    """
    A dictionary containing placeholders meant to be substituted using a
    :class:`ParameterMap`.
    """

    data: dict = attrs.field(factory=dict)

    def render(self, ctx) -> dict:
        """
        Render the template as a nested dictionary using a parameter map to fill
        in empty fields.
        """
        result = self.data.copy()
        skipped = render_params(result, ctx=ctx, flags=ParamFlags.ALL, drop=False)

        # Check for leftover empty values
        if skipped:
            raise ValueError(f"Unevaluated parameters: {skipped}")

        return nest(result, sep=".")


def render_params(
    d: dict, flags: ParamFlags = ParamFlags.ALL, drop: bool = False, **kwargs
) -> list:
    """
    Render parameters in a template dictionary.
    """
    skipped = []

    for k, v in d.items():
        if isinstance(v, Param):
            if v.flags is None:
                skipped.append(k)
            elif v.flags & flags:
                d[k] = v(**kwargs)
            else:
                skipped.append(k)

    if drop:
        for k in skipped:
            del d[k]

    return skipped


# ------------------------------------------------------------------------------
#                           Scene element interface
# ------------------------------------------------------------------------------


class SceneElementFlags(enum.Flag):
    """
    Parameter flags.
    """

    NONE = 0
    COMPOSITE = enum.auto()
    INSTANCE = enum.auto()
    ALL = COMPOSITE | INSTANCE


@attrs.define(eq=False)
class SceneElement:
    """
    Important: All subclasses *must* have a hash, thus eq must be False (see
    attrs docs on hashing for a complete explanation).
    """

    _flags: t.ClassVar[SceneElementFlags] = SceneElementFlags.NONE

    id: t.Optional[str] = documented(
        attrs.field(
            default=None,
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="Identifier of the current scene element.",
        type="str or None",
        init_type="str, optional",
    )

    @classmethod
    @property
    def flags(cls):
        if not cls._flags:
            raise NotImplementedError(
                f"Please specify scene elements flags for '{cls.__name__}'."
            )
        else:
            return cls._flags

    @property
    def kernel_type(self) -> t.Optional[str]:
        """
        Kernel type if this scene element can be modelled by a single kernel
        scene graph node; ``None`` otherwise. The default implementation raises
        a :class:`NotImplementedError`.
        """
        raise NotImplementedError

    @property
    def template(self) -> dict:
        """
        Kernel dictionary template contents associated with this scene element.
        """
        result = {}

        if self.kernel_type is not None:
            result["type"] = self.kernel_type

        if self.params is not None:
            result.update(self.params)

        return result

    @property
    def params(self) -> t.Optional[t.Dict[str, Param]]:
        """
        Map of updatable parameters associated with this scene element.
        """
        return None

    @property
    def objects(self) -> t.Optional[t.Dict[str, SceneElement]]:
        """
        Map of child objects associated with this scene element.
        """
        return None

    @property
    def instance(self) -> t.Optional["mitsuba.Object"]:
        """
        Not None iff element produces no contribution to dictionary template
        and directly expands as a Mitsuba object.
        """
        return None

    def traverse(self, callback: SceneTraversal) -> None:
        """
        Traverse this scene element and collect kernel dictionary template,
        parameter and object map contributions.

        Parameters
        ----------
        callback : SceneTraversal
            Callback data structure storing the collected data.
        """
        try:
            if (
                self.flags & SceneElementFlags.INSTANCE
            ):  # Element expands as instance and doesn't contribute to dict template
                callback.put_instance(self)
            else:  # Element contributes to dict template
                callback.put_template(self.template)

            if self.params is not None:
                callback.put_params(self.params)

            if self.objects is not None:
                for name, obj in self.objects.items():
                    if self.flags & SceneElementFlags.COMPOSITE:
                        template, params = traverse(obj)
                    else:
                        callback.put_object(name, obj)

        except Exception as e:
            raise TraversalError(
                f"Partial callback state: "
                f"depth={callback.depth}, name={callback.name}"
            ) from e


@attrs.define(eq=False)
class Ref(SceneElement):
    id: str = documented(
        attrs.field(
            kw_only=True,
            validator=attrs.validators.instance_of(str),
        ),
        doc="Identifier of the referenced kernel scene object (required).",
        type="str",
    )

    @property
    def kernel_type(self) -> str:
        return "ref"

    @property
    def template(self) -> dict:
        return {**super().template, "id": self.id}


@attrs.define(eq=False)
class Scene(SceneElement):
    _objects: t.Dict[str, SceneElement] = attrs.field(factory=dict, converter=dict)

    @property
    def kernel_type(self) -> str:
        return "scene"

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
    node: SceneElement

    #: Parent to current node
    parent: t.Optional[SceneElement] = attrs.field(default=None)

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

    def put_template(self, template: dict):
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in template.items():
            self.template[f"{prefix}{k}"] = v

    def put_params(self, params: dict):
        prefix = "" if self.name is None else f"{self.name}."

        for k, v in params.items():
            self.params[f"{prefix}{k}"] = v

    def put_object(self, name: str, node: SceneElement):
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
        node.traverse(cb)

    def put_instance(self, obj: "mitsuba.Object"):
        if not self.name:
            raise TraversalError("Instances may only be inserted as child nodes.")
        self.template[self.name] = obj


def traverse(node: SceneElement) -> t.Tuple[KernelDictTemplate, ParameterMap]:
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
