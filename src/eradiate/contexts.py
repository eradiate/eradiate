from __future__ import annotations

import typing as t
import warnings
from abc import ABC, abstractmethod

import attrs
import pint
import pinttr

import eradiate

from . import validators
from .attrs import documented, parse_docs
from .ckd import Bin, Bindex, BinSet
from .exceptions import UnsupportedModeError
from .units import unit_context_config as ucc
from .units import unit_registry as ureg
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
#                               Spectral contexts
# ------------------------------------------------------------------------------


@attrs.frozen
class SpectralContext(ABC, Context):
    """
    Context data structure holding state relevant to the evaluation of
    spectrally dependent objects. This class is abstract.

    This object is usually used as part of a :class:`.KernelDictContext` to pass
    around spectral information to kernel dictionary emission methods which
    require spectral configuration information.

    While this class is abstract, it should however be the main entry point
    to create :class:`.SpectralContext` child class objects through the
    :meth:`.SpectralContext.new` class method constructor.
    """

    @property
    @abstractmethod
    def wavelength(self) -> pint.Quantity:
        """quantity : Wavelength associated with spectral context."""
        # May raise NotImplementedError if irrelevant
        pass

    @property
    @abstractmethod
    def spectral_index(self):
        """Spectral index associated with spectral context."""
        pass

    @property
    @abstractmethod
    def spectral_index_formatted(self) -> str:
        """str : Spectral index formatted as a human-readable string."""
        pass

    @staticmethod
    def new(**kwargs) -> SpectralContext:
        """
        Create a new instance of one of the :class:`SpectralContext` child
        classes. *The instantiated class is defined based on the currently active
        mode.* Keyword arguments are passed to the instantiated class's
        constructor.

        Parameters
        ----------
        **kwargs
            Keyword arguments depending on the currently active mode (see below
            for a list of actual keyword arguments).

        wavelength : quantity or float, default: 550 nm
            (*Monochromatic modes* [:class:`.MonoSpectralContext`])
            Wavelength. Unit-enabled field (default: ucc[wavelength]).

        bindex : .Bindex, optional, default: 1st quadrature point for the "550" \
            bin of the "10nm" bin set (test value)
            (*CKD modes* [:class:`.CKDSpectralContext`])
            CKD bindex.

        bin_set : .BinSet or str or None, optional, default: "10nm" (test value)
            (*CKD modes* [:class:`.CKDSpectralContext`])
            Bin set from which the bindex originates.

        See Also
        --------
        :func:`eradiate.mode`, :func:`eradiate.set_mode`
        """

        if eradiate.mode().is_mono:
            return MonoSpectralContext(**kwargs)

        elif eradiate.mode().is_ckd:
            return CKDSpectralContext(**kwargs)

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @staticmethod
    def from_dict(d: t.Dict) -> SpectralContext:
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameters
        ----------
        d : dict
            Configuration dictionary used for initialisation.

        Returns
        -------
        :class:`.SpectralContext`
            Created object. The actual type depends on context.

        See Also
        --------
        :meth:`.SpectralContext.new`
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return SpectralContext.new(**d_copy)

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.SpectralContext`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return SpectralContext.from_dict(value)

        return value


@parse_docs
@attrs.frozen
class MonoSpectralContext(SpectralContext):
    """
    Monochromatic spectral context data structure.
    """

    _wavelength: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(550.0, ureg.nm),
            units=ucc.deferred("wavelength"),
            on_setattr=None,  # frozen classes can't use on_setattr
        ),
        doc="A single wavelength value.\n\nUnit-enabled field "
        "(default: ucc[wavelength]).",
        type="quantity",
        init_type="quantity or float",
        default="550.0 nm",
    )

    @_wavelength.validator
    def _wavelength_validator(self, attribute, value):
        validators.on_quantity(validators.is_scalar)(self, attribute, value)

    @property
    def wavelength(self):
        """quantity : Wavelength associated with spectral context."""
        return self._wavelength

    @property
    def spectral_index(self) -> float:
        """
        Spectral index associated with spectral context, equal to active
        wavelength magnitude in config units.
        """
        return self._wavelength.m_as(ucc.get("wavelength"))

    @property
    def spectral_index_formatted(self) -> str:
        """str : Formatted spectral index (human-readable string)."""
        return f"{self._wavelength:g~P}"


@parse_docs
@attrs.frozen
class CKDSpectralContext(SpectralContext):
    """
    CKD spectral context data structure.
    """

    bindex: Bindex = documented(
        attrs.field(
            factory=lambda: Bindex(
                BinSet.from_db("10nm").select_bins("550")[0],
                0,
            ),
            converter=Bindex.convert,
            validator=attrs.validators.instance_of(Bindex),
        ),
        doc="The bindex value corresponding to this spectral context. "
        "The default value is a simple placeholder used for testing purposes.",
        type=":class:`.Bindex`",
        init_type=":class:`.Bindex` or tuple or dict, optional",
    )

    bin_set: t.Optional[BinSet] = documented(
        attrs.field(
            default="10nm",
            converter=attrs.converters.optional(BinSet.convert),
            validator=attrs.validators.optional(attrs.validators.instance_of(BinSet)),
        ),
        doc="If relevant, the bin set from which ``bindex`` originates.",
        type=":class:`.BinSet` or None",
        init_type=":class:`.BinSet` or str, optional",
        default='"10nm"',
    )

    @property
    def wavelength(self) -> pint.Quantity:
        """
        quantity : Wavelength associated with spectral context. Alias for \
            ``self.bindex.bin.wcenter``.
        """
        return self.bindex.bin.wcenter

    @property
    def bin(self) -> Bin:
        """
        :class:`.Bin` : Bin associated with spectral context. Alias for \
            ``self.bindex.bin``.
        """
        return self.bindex.bin

    @property
    def spectral_index(self) -> t.Tuple[str, int]:
        """
        tuple[str, int] : Spectral index associated with spectral context, \
            equal to active bindex (bin ID, quadrature point index pair).
        """
        return self.bin.id, self.bindex.index

    @property
    def spectral_index_formatted(self) -> str:
        """str : Formatted spectral index (human-readable string)."""
        return f"{self.bin.id}:{self.bindex.index}"


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

    spectral_ctx: SpectralContext = documented(
        attrs.field(
            factory=SpectralContext.new,
            converter=SpectralContext.convert,
            validator=attrs.validators.instance_of(SpectralContext),
        ),
        doc="Spectral context (used to evaluate quantities with any degree "
        "or kind of dependency vs spectrally varying quantities).",
        type=":class:`.SpectralContext`",
        init_type=":class:`.SpectralContext` or dict",
        default=":meth:`SpectralContext.new() <.SpectralContext.new>`",
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
