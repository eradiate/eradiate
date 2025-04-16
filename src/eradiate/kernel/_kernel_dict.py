"""
This module contains the infrastructure used to generate Mitsuba scene
dictionaries and scene parameter maps.
"""

from __future__ import annotations

import enum
import typing as t
from collections import UserDict
from collections.abc import Mapping

import attrs
import mitsuba as mi

from ..attrs import define, documented
from ..contexts import KernelContext
from ..util.misc import flatten, nest


@define
class DictParameter:
    """
    This class declares an Eradiate parameter in a Mitsuba scene dictionary. It
    holds an evaluation protocol for this parameter depending on context
    information.
    """

    #: Sentinel value indicating that a parameter is not used
    UNUSED: t.ClassVar[object] = object()

    func: t.Callable = documented(
        attrs.field(validator=attrs.validators.is_callable()),
        doc="A callable that returns the value of the parameter for a given "
        "context, with signature ``f(ctx: KernelContext) -> Any``.",
        type="callable",
    )

    def __call__(self, ctx: KernelContext) -> t.Any:
        return self.func(ctx)


class KernelSceneParameterFlags(enum.Flag):
    """A set of flags characterizing the various parameters present in an
    Eradiate scene."""

    NONE = 0
    SPECTRAL = enum.auto()  #: Has a spectral dependence
    GEOMETRIC = (
        enum.auto()
    )  #: Changes the scene geometry and triggers an acceleration structure update
    ALL = SPECTRAL | GEOMETRIC


@define
class SceneParameter:
    """
    This class declares an Eradiate parameter in a Mitsuba scene parameter
    map. It holds an evaluation protocol depending on context information.

    See Also
    --------
    :class:`.KernelContext`, :class:`.SearchSceneParameter`
    """

    #: Sentinel value indicating that a parameter is not used
    UNUSED: t.ClassVar[object] = object()

    func: t.Callable = documented(
        attrs.field(validator=attrs.validators.is_callable()),
        doc="A callable that returns the value of the parameter for a given "
        "context, with signature ``f(ctx: KernelContext) -> Any``.",
        type="callable",
    )

    flags: KernelSceneParameterFlags = documented(
        attrs.field(default=KernelSceneParameterFlags.ALL),
        doc="Flags specifying parameter attributes. By default, the declared "
        "parameter will pass all filters.",
        type=".KernelSceneParameterFlags",
        default=".KernelSceneParameterFlags.ALL",
    )

    search: None | (t.Callable[[mi.Object, str], str | None]) = documented(
        attrs.field(default=None),
        doc="A callable that searches a Mitsuba scene tree node for a desired "
        "parameter ID, with signature "
        "``f(node: mi.Object, node_relpath: str) -> Optional[str]``. This is "
        "typically done with a :class:`.SearchSceneParameter` instance.",
        type="callable or None",
        init_type="callable, optional",
        default="None",
    )

    parameter_id: str | None = documented(
        attrs.field(default=None),
        doc="The full ID of the Mitsuba scene parameter to update.",
        type="str or None",
        init_type="str, optional",
    )

    def __call__(self, ctx: KernelContext) -> t.Any:
        return self.func(ctx)


def dict_parameter(maybe_fn=None):
    """
    This function wraps another one into a :class:`.DictParameter` instance.
    It is primarily meant to be used as a decorator.

    Parameters
    ----------
    maybe_fn : callable, optional

    Returns
    -------
    callable
    """
    return DictParameter if maybe_fn is None else DictParameter(maybe_fn)


def scene_parameter(
    maybe_fn=None,
    *,
    flags: KernelSceneParameterFlags = KernelSceneParameterFlags.ALL,
    node_type: type,
    node_id: str,
    parameter_relpath: str,
):
    """
    This function wraps another one into a :class:`.SceneParameter` instance.
    It is primarily meant to be used as a decorator.

    Parameters
    ----------
    maybe_fn : callable, optional
        A callable that takes as an argument a :class:`.KernelContext` instance.

    flags : .KernelSceneParameterFlags, optional
        Scene parameter flags used for filtering during a scene parameter loop.

    node_type : type
        Type of the node that is expected to hold the parameter updated by the
        wrapped callable.

    node_id : str
        ID of the node that is expected to hold the parameter updated by the
        wrapped callable.

    parameter_relpath : str
        Relative path (from the looked up node) of the parameter updated by the
        wrapped callable.

    Returns
    -------
    callable
    """

    def wrap(f):
        from eradiate.kernel import SearchSceneParameter

        return SceneParameter(
            f,
            flags=flags,
            search=SearchSceneParameter(
                node_type=node_type,
                node_id=node_id,
                parameter_relpath=parameter_relpath,
            ),
        )

    return wrap if maybe_fn is None else wrap(maybe_fn)


@attrs.define(slots=False)
class KernelDict(UserDict):
    """
    A dict-like structure which defines the structure of an instantiable
    Mitsuba scene dictionary.

    Entries are indexed by dot-separated paths which can then be expanded to
    a nested dictionary using the :meth:`.render` method.

    Each entry can be either a hard-coded value which can be directly
    interpreted by the :func:`mitsuba.load_dict` function, or an
    :class:`.DictParameter` object which must be rendered before the template
    can be instantiated.

    Notes
    -----
    If a nested mapping is used for initialization or assignment, it is
    automatically flattened.
    """

    data: dict = attrs.field(factory=dict, converter=flatten)

    def __setitem__(self, key, value):
        if isinstance(value, Mapping):
            value = flatten(value, name=key)
            self.data.update(value)
        else:
            super().__setitem__(key, value)

    def render(
        self, ctx: KernelContext, nested: bool = True, drop: bool = True
    ) -> dict:
        """
        Render the template as a nested dictionary using a parameter map to fill
        in empty fields.

        Parameters
        ----------
        ctx : .KernelContext
            A kernel dictionary context.

        nested : bool, optional
            If ``True``, the returned dictionary will be nested and suitable for
            instantiation by Mitsuba; otherwise, the returned dictionary will be
            flat.

        drop : bool, optional
            If ``True``, drop unused parameters. Parameters may be unused either
            because they were filtered out by the flags or because context
            information implied it.

        Returns
        -------
        dict
        """
        result = {}

        for k, v in list(self.items()):
            value = v(ctx) if isinstance(v, DictParameter) else v
            if (value is DictParameter.UNUSED) and drop:
                continue
            else:
                result[k] = value

        return nest(result, sep=".") if nested else result


@attrs.define(slots=False)
class KernelSceneParameterMap(UserDict):
    """
    A dict-like structure which contains the structure of a Mitsuba scene
    parameter update map.

    Each entry maps a string key to a :class:`.SceneParameter` instance that
    implements an update protocol for a Mitsuba scene parameter.
    """

    data: dict[str, SceneParameter] = attrs.field(factory=dict)

    def render(
        self,
        ctx: KernelContext,
        flags: KernelSceneParameterFlags = KernelSceneParameterFlags.ALL,
        drop: bool = False,
    ) -> dict:
        """
        Evaluate the parameter map for a given kernel context and for selected
        flags.

        Parameters
        ----------
        ctx : :class:`.KernelContext`
            A kernel dictionary context.

        flags : :class:`.ParamFlags`
            Parameter flags. Only parameters with at least one of the specified
            will pass the filter.

        drop : bool, optional
            If ``True``, drop unused parameters. Parameters may be unused either
            because they were filtered out by the flags or because context
            information implied it. If ``False``, any unused parameter will
            raise an exception.

        Returns
        -------
        params : dict
            A mapping that can be used to update values in a
            :class:`mitsuba.SceneParameters`.

        Raises
        ------
        ValueError
            If a value is not a :class:`.SceneParameter`.

        ValueError
            If ``drop`` is ``False`` and the rendered parameter map contains an
            unused parameter.
        """
        unused = []
        result = {}

        for k in list(self.keys()):
            v = self[k]

            if isinstance(v, SceneParameter):
                key = k if v.parameter_id is None else v.parameter_id

                if v.flags & flags:
                    result[key] = v(ctx)
                else:
                    unused.append(k)
                    if not drop:
                        result[key] = SceneParameter.UNUSED

        # Check for leftover empty values
        if not drop and unused:
            raise ValueError(f"Unevaluated parameters: {unused}")

        return result
