"""
This module contains the infrastructure used to generate Mitsuba scene
dictionaries and scene parameter maps.
"""

from __future__ import annotations

import enum
import typing as t
from collections import UserDict

import attrs
import mitsuba as mi

from eradiate.attrs import documented, parse_docs
from eradiate.contexts import KernelDictContext
from eradiate.util.misc import nest


class ParamFlags(enum.Flag):
    """
    Parameter flags.
    """

    NONE = 0
    SPECTRAL = enum.auto()  #: Varies during the spectral loop
    GEOMETRIC = enum.auto()  #: Triggers a scene rebuild
    ALL = SPECTRAL | GEOMETRIC


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


@parse_docs
@attrs.define
class Parameter:
    """
    This class declares an Eradiate parameter. It holds an evaluation protocol
    for a Mitsuba scene parameter and exposes an interface to generate update
    values depending on context information.

    Notes
    -----
    This class cannot be instantiated directly. Instead, its children
    :class:`.InitParameter` and :class:`.UpdateParameter` should be used.
    """

    evaluator: t.Callable = documented(
        attrs.field(repr=False),
        doc="A callable that returns the value of the parameter for a given "
        "context, with signature ``f(ctx: KernelDictContext) -> Any``.",
        type="callable",
    )

    def __attrs_pre_init__(self):
        raise TypeError(
            f"cannot instantiate abstract class '{self.__class__.__name__}'"
        )

    def __call__(self, ctx: KernelDictContext) -> t.Any:
        return self.evaluator(ctx)


@parse_docs
@attrs.define
class InitParameter(Parameter):
    """
    This class declares an Eradiate parameter in a kernel dictionary. It holds
    an evaluation protocol for a Mitsuba scene parameter and exposes an
    interface to generate update values depending on context information.
    """

    def __attrs_pre_init__(self):
        pass


@attrs.define
class UpdateParameter(Parameter):
    """
    This class declares an Eradiate parameter in a scene parameter update map.
    It holds an evaluation protocol for a Mitsuba scene parameter and exposes an
    interface to generate update values depending on context information.
    """

    #: Sentinel value indicated that a parameter is not used
    UNUSED: t.ClassVar = _Unused()

    flags: ParamFlags = documented(
        attrs.field(default=ParamFlags.ALL),
        doc="Flags specifying parameter attributes. By default, the declared "
        "parameter will pass all filters.",
        type=".ParaFlags",
        default=".ParamFlags.ALL",
    )

    lookup_id: t.Optional[t.Callable[[mi.Object, str], t.Optional[str]]] = documented(
        attrs.field(default=None),
        doc="A callable that searches a Mitsuba scene tree node for a desired "
        "parameter ID.",
        type="callable or None",
        init_type="callable, optional",
        default="None",
    )

    parameter_id: t.Optional[str] = documented(
        attrs.field(default=None),
        doc="The full ID of the Mitsuba scene parameter to update.",
        type="str or None",
        init_type="str, optional",
    )

    def __attrs_pre_init__(self):
        pass


@attrs.define(slots=False)
class KernelDictTemplate(UserDict):
    """
    A dict-like structure which contains the structure of an instantiable kernel
    dictionary.

    Entries are indexed by dot-separated paths which can then be expanded to
    a nested dictionary using the :meth:`.render` method.

    Each entry can be either a hard-coded value which can be directly
    interpreted by the :func:`mitsuba.load_dict` function, or an
    :class:`.InitParameter` object, which must be rendered before the template
    can be instantiated.
    """

    data: dict[str, Parameter] = attrs.field(factory=dict)

    def render(self, ctx: KernelDictContext, nested: bool = True) -> dict:
        """
        Render the template as a nested dictionary using a parameter map to fill
        in empty fields.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A kernel dictionary context.

        nested : bool, optional
            If ``True``, the returned dictionary will be nested and suitable for
            instantiation by Mitsuba; otherwise, the returned dictionary will be
            flat.

        Returns
        -------
        dict
        """
        result = {}

        for k, v in list(self.items()):
            result[k] = v(ctx) if isinstance(v, InitParameter) else v

        return nest(result, sep=".") if nested else result


@attrs.define(slots=False)
class UpdateMapTemplate(UserDict):
    """
    A dict-like structure which contains the structure of a Mitsuba scene
    parameter update map.

    Entries are indexed by dot-separated paths which can then be expanded to
    a nested dictionary using the :meth:`.render` method.
    """

    data: dict[str, Parameter] = attrs.field(factory=dict)

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

        drop : bool, optional
            If ``True``, drop unused parameters. Parameters may be unused either
            because they were filtered out by the flags or because context
            information implied it.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If a value is not a :class:`.UpdateParameter`.

        ValueError
            If ``drop`` is ``False`` and the rendered parameter map contains an
            unused parameter.
        """
        unused = []
        result = {}

        for k in list(self.keys()):
            v = self[k]

            if isinstance(v, UpdateParameter):
                key = k if v.parameter_id is None else v.parameter_id

                if v.flags & flags:
                    result[key] = v(ctx)
                else:
                    unused.append(k)
                    if not drop:
                        result[key] = UpdateParameter.UNUSED

        # Check for leftover empty values
        if not drop and unused:
            raise ValueError(f"Unevaluated parameters: {unused}")

        return result
