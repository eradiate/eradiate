from __future__ import annotations

import enum
import functools
import typing as t

import attrs
import mitsuba

from .attrs import documented, frozen
from .exceptions import UnsetModeError, UnsupportedModeError

# ------------------------------------------------------------------------------
#                                 Mode flags
# ------------------------------------------------------------------------------


class ModeFlag(enum.Flag):
    """
    Flags defining the various possible features associated to an Eradiate mode.
    """

    NONE = 0  #: No flags
    SPECTRAL_MODE_MONO = enum.auto()  #: Monochromatic (line-by-line) mode
    SPECTRAL_MODE_CKD = enum.auto()  #: Correlated-k distribution mode
    MI_BACKEND_SCALAR = enum.auto()  #: Mitsuba scalar backend
    MI_BACKEND_LLVM = enum.auto()  #: Mitsuba LLVM backend
    MI_COLOR_MODE_MONO = enum.auto()  #: Mitsuba monochromatic mode
    MI_DOUBLE_PRECISION_NO = enum.auto()  #: Mitsuba single-precision
    MI_DOUBLE_PRECISION_YES = enum.auto()  #: Mitsuba double-precision
    MI_POLARIZED_NO = enum.auto()  #: Unpolarized
    MI_POLARIZED_YES = enum.auto()  #: Polarized
    ANY = (
        SPECTRAL_MODE_MONO
        | SPECTRAL_MODE_CKD
        | MI_BACKEND_SCALAR
        | MI_BACKEND_LLVM
        | MI_COLOR_MODE_MONO
        | MI_DOUBLE_PRECISION_NO
        | MI_DOUBLE_PRECISION_YES
        | MI_POLARIZED_NO
        | MI_POLARIZED_YES
    )  #: All flags


# ------------------------------------------------------------------------------
#                              Mode definitions
# ------------------------------------------------------------------------------


# Map associating a mode ID string to the corresponding class
# (aliased in public API section). This is implemented as a cached function to
# make a delayed evaluation possible (otherwise, the Mode class is not defined).
# See also the Mode.new() constructor.
@functools.lru_cache(maxsize=1)
def _mode_registry() -> dict[str, Mode]:
    return {
        k: Mode(id=k, **v)
        for k, v in {
            "mono_single": {
                "spectral_mode": "mono",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": False,
                "mi_polarized": False,
            },
            "mono_polarized_single": {
                "spectral_mode": "mono",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": False,
                "mi_polarized": True,
            },
            "mono_double": {
                "spectral_mode": "mono",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": True,
                "mi_polarized": False,
            },
            "mono_polarized_double": {
                "spectral_mode": "mono",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": True,
                "mi_polarized": True,
            },
            "ckd_single": {
                "spectral_mode": "ckd",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": False,
                "mi_polarized": False,
            },
            "ckd_polarized_single": {
                "spectral_mode": "ckd",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": False,
                "mi_polarized": True,
            },
            "ckd_double": {
                "spectral_mode": "ckd",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": True,
                "mi_polarized": False,
            },
            "ckd_polarized_double": {
                "spectral_mode": "ckd",
                "mi_backend": "scalar",
                "mi_color_mode": "mono",
                "mi_double_precision": True,
                "mi_polarized": True,
            },
        }.items()
    }


def _spectral_mode_converter(value: str | ModeFlag):
    if isinstance(value, str):
        value = value.upper()
        if not value.startswith("SPECTRAL_MODE_"):
            value = f"SPECTRAL_MODE_{value}"
        return ModeFlag[value]
    else:
        return ModeFlag(value)


def _mi_backend_converter(value: str | ModeFlag):
    if isinstance(value, str):
        value = value.upper()
        if not value.startswith("MI_BACKEND_"):
            value = f"MI_BACKEND_{value}"
        return ModeFlag[value]
    else:
        return ModeFlag(value)


def _mi_color_mode_converter(value: str | ModeFlag):
    if isinstance(value, str):
        value = value.upper()
        if not value.startswith("MI_COLOR_MODE_"):
            value = f"MI_COLOR_MODE_{value}"
        return ModeFlag[value]
    else:
        return ModeFlag(value)


def _mi_polarized_converter(value: bool | ModeFlag):
    if isinstance(value, bool):
        return ModeFlag.MI_POLARIZED_YES if value is True else ModeFlag.MI_POLARIZED_NO
    else:
        return ModeFlag(value)


def _mi_double_precision_converter(value: bool | ModeFlag):
    if isinstance(value, bool):
        return (
            ModeFlag.MI_DOUBLE_PRECISION_YES
            if value is True
            else ModeFlag.MI_DOUBLE_PRECISION_NO
        )
    else:
        return ModeFlag(value)


@frozen
class Mode:
    """
    Data structure describing Eradiate's operational mode and associated
    ancillary data.

    Warnings
    --------
    Instances are immutable.
    """

    id: str = documented(
        attrs.field(converter=attrs.converters.optional(str)),
        doc="Mode identifier.",
        type="str",
    )

    spectral_mode: t.Literal[
        ModeFlag.SPECTRAL_MODE_MONO, ModeFlag.SPECTRAL_MODE_CKD
    ] = documented(
        attrs.field(
            converter=_spectral_mode_converter,
            validator=attrs.validators.in_(
                {ModeFlag.SPECTRAL_MODE_MONO, ModeFlag.SPECTRAL_MODE_CKD}
            ),
        ),
        doc="Spectral dimension handling.",
        type=".ModeFlag",
        init_type=".ModeFlag or str",
    )

    mi_backend: t.Literal[ModeFlag.MI_BACKEND_SCALAR, ModeFlag.MI_BACKEND_LLVM] = (
        documented(
            attrs.field(
                converter=_mi_backend_converter,
                validator=attrs.validators.in_(
                    {ModeFlag.MI_BACKEND_SCALAR, ModeFlag.MI_BACKEND_LLVM}
                ),
            ),
            doc="Mitsuba computational backend.",
            type=".ModeFlag",
            init_type=".ModeFlag or str",
        )
    )

    mi_color_mode: t.Literal[ModeFlag.MI_COLOR_MODE_MONO] = documented(
        attrs.field(
            converter=_mi_color_mode_converter,
            validator=attrs.validators.in_({ModeFlag.MI_COLOR_MODE_MONO}),
        ),
        doc="Mitsuba color mode.",
        type=".ModeFlag",
        init_type=".ModeFlag or str",
    )

    mi_polarized: t.Literal[ModeFlag.MI_POLARIZED_NO, ModeFlag.MI_POLARIZED_YES] = (
        documented(
            attrs.field(
                converter=_mi_polarized_converter,
                validator=attrs.validators.in_(
                    {ModeFlag.MI_POLARIZED_NO, ModeFlag.MI_POLARIZED_YES}
                ),
            ),
            doc="Mitsuba polarized mode.",
            type=".ModeFlag",
            init_type=".ModeFlag or str",
        )
    )

    mi_double_precision: t.Literal[
        ModeFlag.MI_DOUBLE_PRECISION_NO, ModeFlag.MI_DOUBLE_PRECISION_YES
    ] = documented(
        attrs.field(
            converter=_mi_double_precision_converter,
            validator=attrs.validators.in_(
                {ModeFlag.MI_DOUBLE_PRECISION_NO, ModeFlag.MI_DOUBLE_PRECISION_YES}
            ),
        ),
        doc="Mitsuba double precision.",
        type=".ModeFlag",
        init_type=".ModeFlag or str",
    )

    @property
    def mi_flags(self) -> ModeFlag:
        """
        Combined Mitsuba-specific flags.
        """
        return (
            self.mi_backend
            | self.mi_color_mode
            | self.mi_polarized
            | self.mi_double_precision
        )

    @property
    def flags(self) -> ModeFlag:
        """
        All flags combined.
        """
        return self.spectral_mode | self.mi_flags

    @property
    def mi_variant(self) -> str:
        """
        Mitsuba variant associated with the selected mode.
        """
        result = [
            self.mi_backend.name.lower().removeprefix("mi_backend_"),
            self.mi_color_mode.name.lower().removeprefix("mi_color_mode_"),
        ]
        if self.mi_polarized is ModeFlag.MI_POLARIZED_YES:
            result.append("polarized")
        if self.mi_double_precision is ModeFlag.MI_DOUBLE_PRECISION_YES:
            result.append("double")
        return "_".join(result)

    def check(
        self,
        spectral_mode: ModeFlag | str = ModeFlag.NONE,
        mi_backend: ModeFlag | str = ModeFlag.NONE,
        mi_color_mode: ModeFlag | str = ModeFlag.NONE,
        mi_polarized: ModeFlag | bool = ModeFlag.NONE,
        mi_double_precision: ModeFlag | bool = ModeFlag.NONE,
    ) -> ModeFlag:
        """
        Check if the currently active mode has the passed flags.

        Parameters
        ----------
        spectral_mode : .ModeFlag or str, optional
            Spectral mode to check. If unset, the check is skipped.

        mi_backend : .ModeFlag or str, optional
            Mitsuba backend to check. If unset, the check is skipped.

        mi_color_mode : .ModeFlag or str, optional
            Mitsuba color mode to check. If unset, the check is skipped.

        mi_polarized : .ModeFlag or bool, optional
            Mitsuba polarized mode to check. If unset, the check is skipped.

        mi_double_precision : .ModeFlag or bool, optional
            Mitsuba double precision mode to check. If unset, the check is skipped.

        Returns
        -------
        bool
            ``True`` if current mode has the passed flags, ``False`` otherwise.
        """
        condition = (
            _spectral_mode_converter(spectral_mode)
            | _mi_backend_converter(mi_backend)
            | _mi_color_mode_converter(mi_color_mode)
            | _mi_polarized_converter(mi_polarized)
            | _mi_double_precision_converter(mi_double_precision)
        )
        return condition & self.flags

    @property
    def is_mono(self) -> bool:
        return self.spectral_mode is ModeFlag.SPECTRAL_MODE_MONO

    @property
    def is_ckd(self) -> bool:
        return self.spectral_mode is ModeFlag.SPECTRAL_MODE_CKD

    @property
    def is_single_precision(self) -> bool:
        return self.mi_double_precision is ModeFlag.MI_DOUBLE_PRECISION_NO

    @property
    def is_double_precision(self) -> bool:
        return self.mi_double_precision is ModeFlag.MI_DOUBLE_PRECISION_YES

    @property
    def is_polarized(self) -> bool:
        return self.mi_polarized is ModeFlag.MI_POLARIZED_YES

    @staticmethod
    def new(mode_id: str) -> Mode:
        """
        Create a :class:`Mode` instance given its identifier. Available modes are:

        * ``mono``: Alias to ``mono_double``
        * ``ckd``: Alias to ``ckd_double``
        * ``mono_polarized``: Alias to ``mono_polarized_double``
        * ``ckd_polarized``: Alias to ``ckd_polarized_double``
        * ``mono_single``: Monochromatic, single-precision
        * ``mono_polarized_single``: Monochromatic, polarized, single-precision
        * ``mono_double``: Monochromatic, double-precision
        * ``mono_polarized_double``: Monochromatic, polarized, double-precision
        * ``ckd_single``: CKD, single-precision
        * ``ckd_polarized_single``: CKD, polarized, single-precision
        * ``ckd_double``: CKD, double-precision
        * ``ckd_polarized_double``: CKD, polarized, double-precision

        Parameters
        ----------
        mode_id : str
            String identifier for the created :class:`Mode` instance.

        Returns
        -------
        .Mode
            Created :class:`.Mode` instance.
        """
        try:
            return _mode_registry()[mode_id]
        except KeyError:
            raise ValueError(f"unknown mode '{mode_id}'")


# Define mode aliases
_mode_registry().update(
    {
        "mono": _mode_registry()["mono_double"],
        "mono_polarized": _mode_registry()["mono_polarized_double"],
        "ckd": _mode_registry()["ckd_double"],
        "ckd_polarized": _mode_registry()["ckd_polarized_double"],
    }
)

# Eradiate's operational mode configuration
_active_mode: Mode | None = None


# ------------------------------------------------------------------------------
#                            Mode subtype dispatcher
# ------------------------------------------------------------------------------


@attrs.define
class SubtypeDispatcher:
    """
    This is a very simple factory intended to map mode-specific flags to
    mode-dependent subtypes.
    Types can be registered with the :meth:`.register` method, and the
    appropriate subtype can be resolved with the :meth:`.resolve` method.
    """

    _type_name: str = attrs.field()
    _registry: dict[ModeFlag, t.Type] = attrs.field(factory=dict)

    def register(self, mode_flags: ModeFlag | str) -> t.Callable[[t.Type], t.Type]:
        """
        Register a subtype against a combination of mode flags. This method is
        meant to be used as a decorator.

        Parameters
        ----------
        mode_flags : .ModeFlag or str
            Mode flags against which the subtype is registered. If a string is
            passed, it is converted to a :class:`.ModeFlag`.
        """
        if isinstance(mode_flags, str):
            mode_flags = ModeFlag[mode_flags.upper()]

        def wrapper(cls):
            self._registry[mode_flags] = cls
            return cls

        return wrapper

    def resolve(self, mode_flags: ModeFlag | None = None) -> t.Type:
        """
        Resolve the subtype based against a set of mode flags.

        Parameters
        ----------
        mode_flags : .ModeFlag, optional
            A mode flag combination used to search the dispatcher's registry.
            The first entry that validates the flag conditions is returned,
            meaning that conflicting or redundant conditions will cause issues.
            If unspecified, the flags of the currently active mode are used.

        See Also
        --------
        :attr:`.Mode.flags`, :class:`.ModeFlag`
        """
        active_mode = get_mode()

        if mode_flags is None:
            mode_flags = active_mode.flags

        for key, value in self._registry.items():
            if mode_flags & key:
                return value

        raise NotImplementedError(
            f"Type {self._type_name} has no registered subtype for mode flags "
            f"{mode_flags}."
        )


# ------------------------------------------------------------------------------
#                                Public API
# ------------------------------------------------------------------------------


def get_mode(raise_exc: bool = True) -> Mode | None:
    """
    Get current operational mode.

    Parameters
    ----------
    raise_exc : bool, default: True
        If ``True``, raise an exception if mode has not been set prior
        (otherwise, this function will return ``None`` in such cases).

    Returns
    -------
    .Mode or None
        Current operational mode.

    Raises
    ------
    UnsetModeError
        If no active mode has been selected.

    See Also
    --------
    :func:`set_mode`
    """
    if _active_mode is None and raise_exc is True:
        raise UnsetModeError
    return _active_mode


def mode(raise_exc: bool = True) -> Mode | None:
    """This is a thin compatibility wrapper around :func:`.get_mode()`."""
    return get_mode(raise_exc=raise_exc)


def modes(
    filter: t.Callable[[Mode], bool] | None = None, asdict: bool = False
) -> list[str] | dict[str, Mode]:
    """
    Get list of registered operational modes.

    Parameters
    ----------
    filter : callable, optional
        A callable used to filter the returned modes. Operates on a
        :class:`.Mode` instance.

    asdict : bool, default: False
        If ``True``, returns a mode ID → mode instance dictionary; otherwise,
        returns a list of mode IDs only.

    Returns
    -------
    modes : list[str] or dict[str, .Mode]
        List of registered operational modes.

    Examples
    --------
    Return the full list of registered modes:

    >>> eradiate.modes()

    Return only CKD modes:

    >>> eradiate.modes(lambda x: x.is_ckd)
    """
    if filter is None:
        filter = lambda x: True  # noqa: E731

    if not asdict:
        return [k for k, v in _mode_registry().items() if filter(v)]
    else:
        return {k: v for k, v in _mode_registry().items() if filter(v)}


def set_mode(mode_id: str):
    """
    Set Eradiate's operational mode.

    This function sets and configures Eradiate's operational mode. Eradiate's
    modes map to Mitsuba's variants and are used to make contextual decisions
    when relevant during the translation of a scene to its kernel format.

    .. admonition:: Valid mode IDs
       :class: info

       * ``mono``: Alias to ``mono_double``
       * ``ckd``: Alias to ``ckd_double``
       * ``mono_polarized``: Alias to ``mono_polarized_double``
       * ``ckd_polarized``: Alias to ``ckd_polarized_double``
       * ``mono_single``: Monochromatic, single-precision
       * ``mono_polarized_single``: Monochromatic, polarized, single-precision
       * ``mono_double``: Monochromatic, double-precision
       * ``mono_polarized_double``: Monochromatic, polarized, double-precision
       * ``ckd_single``: CKD, single-precision
       * ``ckd_polarized_single``: CKD, polarized, single-precision
       * ``ckd_double``: CKD, double-precision
       * ``ckd_polarized_double``: CKD, polarized, double-precision
       * ``none`` (no mode selected)

    Parameters
    ----------
    mode_id : str
        Mode to be selected (see list below).

    Raises
    ------
    ValueError
        ``mode_id`` does not match any of the known mode identifiers.
    """
    global _active_mode

    if mode_id in _mode_registry():
        mode = Mode.new(mode_id)
        mitsuba.set_variant(mode.mi_variant)
    elif mode_id.lower() == "none":
        mode = None
    else:
        raise ValueError(f"unknown mode '{mode_id}'")

    _active_mode = mode


def supported_mode(**kwargs):
    """
    Check whether the current mode has specific features. If not, raise.

    Parameters
    ----------
    kwargs
        Keyword arguments passed to :meth:`.Mode.check`.

    Raises
    ------
    UnsupportedModeError
        Current mode does not pass the check.
    """
    if mode() is None or not mode().check(**kwargs):
        raise UnsupportedModeError


def unsupported_mode(**kwargs):
    """
    Check whether the current mode has specific features. If so, raise.

    Parameters
    ----------
    kwargs
        Keyword arguments passed to :meth:`.Mode.check`.

    Raises
    ------
    UnsupportedModeError
        Current mode has the requested flags.
    """
    if mode() is None or mode().check(**kwargs):
        raise UnsupportedModeError
