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
@attrs.define
class KernelDictContext(Context):
    """
    Kernel dictionary evaluation context data structure. This class is used
    *e.g.* to store information about the spectral configuration to apply
    when generating kernel dictionaries associated with a :class:`.SceneElement`
    instance.
    """

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

    kwargs: t.Dict[str, t.Any] = documented(
        attrs.field(factory=dict),
        doc="Object-specific parameter overrides.",
        type="dict",
        default="{}",
    )
