from __future__ import annotations

import typing as t
import warnings

import attrs

from .attrs import documented, parse_docs
from .spectral_index import SpectralIndex
from .util.misc import fullname

# ------------------------------------------------------------------------------
#                                      ABC
# ------------------------------------------------------------------------------


@attrs.define
class Context:
    """Abstract base class for all context data structures."""

    def evolve(self, **changes):
        """
        Create a copy of self with changes applied.

        Parameters
        ----------
        **changes
            Keyword changes in the new copy.

        Returns
        -------
        <same type as self>
            A copy of self with ``changes`` incorporated.
        """
        return attrs.evolve(self, **changes)

# ------------------------------------------------------------------------------
#                         Kernel dictionary contexts
# ------------------------------------------------------------------------------


@parse_docs
@attrs.frozen(init=False)
class KernelDictContext(Context):
    """
    Kernel dictionary evaluation context data structure. This class is used
    *e.g.* to store information about the spectral configuration to apply
    when generating kernel dictionaries associated with a :class:`.SceneElement`
    instance.

    A number of static fields are defined and automatically initialised upon
    instantiation. In addition, the `dynamic` field stores a dictionary in
    which arbitrary data can be stored.

    Both the static and dynamic data can be accessed as attributes. However,
    since instances are frozen, the recommended way to update a
    :class:`KernelDictContext` is to use its :meth:`evolve` method and create
    a copy.

    Although adding just any dynamic field to a :class:`KernelDictContext`
    instance is possible, this class also holds a registry of known dynamic
    fields (see :ref:`sec-contexts-context_fields`).

    Notes
    -----
    In the unlikely (pointless but technically possible) event of a dynamic data
    field having the same name as a static field, the former takes precedence
    when using attribute access.

    Examples
    --------
    In addition to the regular initialisation pattern (*i.e.* passing field
    values as arguments to the constructor), the `dynamic` dictionary can also be
    populated by passing additional keyword arguments. For instance, the two
    following init patterns are equivalent:

    .. testsetup:: contexts

       import eradiate
       from eradiate.contexts import KernelDictContext
       eradiate.set_mode("mono")

    .. doctest:: contexts
       :options: +ELLIPSIS

       >>> KernelDictContext(dynamic={"foo": "bar"})
       KernelDictContext(..., dynamic={'foo': 'bar'})
       >>> KernelDictContext(foo="bar")
       KernelDictContext(..., dynamic={'foo': 'bar'})

    Access to dynamic data can go through direct query of the dictionary, or
    using attributes:

    .. doctest:: contexts

       >>> ctx = KernelDictContext(foo="bar")
       >>> ctx.dynamic["foo"]
       'bar'
       >>> ctx.foo
       'bar'

    .. testcleanup:: contexts

       del KernelDictContext
    """

    @parse_docs
    @attrs.define
    class DynamicFieldRegistry:
        """
        Record a list of dynamic fields.
        """

        data: t.Dict[str, t.List[str]] = documented(
            attrs.field(factory=dict),
            doc="Registry data. Each entry associates a dynamic field name to "
            "the fully qualified name of the function which registered it.",
            type="dict",
            default="{}",
        )

        def register(self, *field_names):
            """
            This decorator can be used to annotate a function and register one
            or several dynamic fields.
            """

            def decorator(func):
                for field_name in field_names:
                    if field_name in self.data:
                        self.data[field_name].append(fullname(func))
                    else:
                        self.data[field_name] = [fullname(func)]

                return func

            return decorator

        def registered(self, field_name):
            try:
                return self.data[field_name]
            except KeyError:
                return None

    #: Class instance of :class:`DynamicFieldRegistry` which keeps track of
    #: dynamic context fields declared by functions in Eradiate.
    DYNAMIC_FIELDS = DynamicFieldRegistry()

    spectral_index: SpectralIndex = documented(
        attrs.field(
            factory=SpectralIndex.new,
            converter=SpectralIndex.convert,
            validator=attrs.validators.instance_of(SpectralIndex),
        ),
        doc="Spectral index.",
        type=":class:`.SpectralIndex`",
        init_type=":class:`.SpectralIndex` or dict",
        default=":meth:`SpectralIndex.new() <.SpectralIndex.new>`",
    )

    ref: bool = documented(
        attrs.field(default=True, converter=bool),
        doc="If ``True``, use references when relevant during kernel dictionary "
        "generation.",
        type="bool",
        default="True",
    )

    dynamic: dict = documented(
        attrs.field(factory=dict),
        doc="A dynamic table containing specific context data.",
        type="dict",
        default="{}",
    )

    @dynamic.validator
    def _dynamic_validator(self, attribute, value):
        for field in value:
            if field not in self.DYNAMIC_FIELDS.data:
                warnings.warn(f"Dynamic field '{field}' was set but is not used")

    def __init__(self, **kwargs):
        fields = {
            field.name: field.default.factory()
            if isinstance(field.default, attrs.Factory)
            else field.default
            for field in self.__attrs_attrs__
        }

        for key, value in kwargs.items():
            if key in fields:
                if key != "dynamic":
                    fields[key] = value
                else:
                    fields[key].update(value)
            else:
                fields["dynamic"][key] = value

        self.__attrs_init__(**fields)

    def __getattr__(self, k):
        """
        Gets key if it exists, otherwise throws AttributeError.
        nb. __getattr__ is only called if key is not found in normal places.
        """
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self.dynamic[k]
            except KeyError:
                raise AttributeError(k)

    def evolve(self, **changes):
        """
        Create a copy of self with changes applied. As for the constructor,
        updates to the dynamic table can be passed as keyword arguments. This is
        the recommended way.
        """
        # Discriminate static and dynamic items
        static = {}
        dynamic = {}

        for key, value in changes.items():
            try:
                self.__getattribute__(key)
                static[key] = value
            except AttributeError:
                dynamic[key] = value

        return attrs.evolve(self, **static, dynamic={**self.dynamic, **dynamic})
