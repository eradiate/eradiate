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

from ..attrs import documented, parse_docs
from ..contexts import KernelContext
from ..util.misc import nest


@parse_docs
@attrs.define
class InitParameter:
    """
    This class declares an Eradiate parameter in a Mitsuba scene dictionary. It
    holds an evaluation protocol for this parameter depending on context
    information.
    """

    #: Sentinel value indicating that a parameter is not used
    UNUSED: t.ClassVar[object] = object()

    evaluator: t.Callable = documented(
        attrs.field(validator=attrs.validators.is_callable()),
        doc="A callable that returns the value of the parameter for a given "
        "context, with signature ``f(ctx: KernelContext) -> Any``.",
        type="callable",
    )

    def __call__(self, ctx: KernelContext) -> t.Any:
        return self.evaluator(ctx)


@parse_docs
@attrs.define
class UpdateParameter:
    """
    This class declares an Eradiate parameter in a Mitsuba scene parameter
    update map. It holds an evaluation protocol depending on context
    information.

    See Also
    --------
    :class:`.KernelContext`, :class:`.TypeIdLookupStrategy`
    """

    #: Sentinel value indicating that a parameter is not used
    UNUSED: t.ClassVar[object] = object()

    class Flags(enum.Flag):
        """
        Update parameter flags.
        """

        NONE = 0
        SPECTRAL = enum.auto()  #: Varies during the spectral loop
        GEOMETRIC = enum.auto()  #: Triggers a scene rebuild
        ALL = SPECTRAL | GEOMETRIC

    evaluator: t.Callable = documented(
        attrs.field(validator=attrs.validators.is_callable()),
        doc="A callable that returns the value of the parameter for a given "
        "context, with signature ``f(ctx: KernelContext) -> Any``.",
        type="callable",
    )

    flags: Flags = documented(
        attrs.field(default=Flags.ALL),
        doc="Flags specifying parameter attributes. By default, the declared "
        "parameter will pass all filters.",
        type=".Flags",
        default=".Flags.ALL",
    )

    lookup_strategy: None | (t.Callable[[mi.Object, str], str | None]) = documented(
        attrs.field(default=None),
        doc="A callable that searches a Mitsuba scene tree node for a desired "
        "parameter ID: with signature "
        "``f(node: mi.Object, node_relpath: str) -> Optional[str]``.",
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
        return self.evaluator(ctx)


@attrs.define(slots=False)
class KernelDictTemplate(UserDict):
    """
    A dict-like structure which defines the structure of an instantiable
    Mitsuba scene dictionary.

    Entries are indexed by dot-separated paths which can then be expanded to
    a nested dictionary using the :meth:`.render` method.

    Each entry can be either a hard-coded value which can be directly
    interpreted by the :func:`mitsuba.load_dict` function, or an
    :class:`.InitParameter` object which must be rendered before the template
    can be instantiated.
    """

    data: dict[str, InitParameter] = attrs.field(factory=dict)

    def render(
        self, ctx: KernelContext, nested: bool = True, drop: bool = True
    ) -> dict:
        """
        Render the template as a nested dictionary using a parameter map to fill
        in empty fields.

        Parameters
        ----------
        ctx : :class:`.KernelContext`
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
            value = v(ctx) if isinstance(v, InitParameter) else v
            if (value is InitParameter.UNUSED) and drop:
                continue
            else:
                result[k] = value

        return nest(result, sep=".") if nested else result


@attrs.define(slots=False)
class UpdateMapTemplate(UserDict):
    """
    A dict-like structure which contains the structure of a Mitsuba scene
    parameter update map.

    Entries are indexed by dot-separated paths which can then be expanded to
    a nested dictionary using the :meth:`.render` method.
    """

    data: dict[str, UpdateParameter] = attrs.field(factory=dict)

    def remove(self, keys: str | list[str]) -> None:
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

    def keep(self, keys: str | list[str]) -> None:
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
        ctx: KernelContext,
        flags: UpdateParameter.Flags = UpdateParameter.Flags.ALL,
        drop: bool = False,
    ) -> dict:
        """
        Evaluate the parameter map for a set of arguments.

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

        for k in list(
            self.keys()
        ):  # Ensures correct iteration even if the loop mutates the mapping
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
