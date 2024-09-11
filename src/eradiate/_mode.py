from __future__ import annotations

import enum
import functools
import typing as t

import attrs
import mitsuba

from .attrs import documented, frozen
from .exceptions import UnsupportedModeError

# ------------------------------------------------------------------------------
#                                 Mode flags
# ------------------------------------------------------------------------------


class ModeFlag(enum.Flag):
    NONE = 0
    SPECTRAL_MODE_MONO = enum.auto()  #: Monochromatic (line-by-line) mode
    SPECTRAL_MODE_CKD = enum.auto()  #: Correlated-k distribution mode
    MI_BACKEND_SCALAR = enum.auto()  #: Scalar backend
    MI_BACKEND_LLVM = enum.auto()  #: LLVM backend
    MI_COLOR_MODE_MONO = enum.auto()  #: Monochromatic mode
    MI_DOUBLE_PRECISION_NO = enum.auto()
    MI_DOUBLE_PRECISION_YES = enum.auto()
    MI_POLARIZED_NO = enum.auto()
    MI_POLARIZED_YES = enum.auto()
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
    )


_SPECTRAL_COORD_LABELS = {
    ModeFlag.SPECTRAL_MODE_MONO: "w",
    ModeFlag.SPECTRAL_MODE_CKD: "bd",
}


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
                "spectral_mode": SpectralMode.MONO,
                "mi_backend": MitsubaBackend.SCALAR,
                "mi_color_mode": MitsubaColorMode.MONO,
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
                "spectral_mode": SpectralMode.MONO,
                "mi_backend": MitsubaBackend.SCALAR,
                "mi_color_mode": MitsubaColorMode.MONO,
                "mi_double_precision": True,
                "mi_polarized": True,
            },
            "mono_polarized": {
                "spectral_mode": SpectralMode.MONO,
                "mi_backend": MitsubaBackend.SCALAR,
                "mi_color_mode": MitsubaColorMode.MONO,
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
                "spectral_mode": SpectralMode.CKD,
                "mi_backend": MitsubaBackend.SCALAR,
                "mi_color_mode": MitsubaColorMode.MONO,
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
                "spectral_mode": SpectralMode.CKD,
                "mi_backend": MitsubaBackend.SCALAR,
                "mi_color_mode": MitsubaColorMode.MONO,
                "mi_double_precision": True,
                "mi_polarized": True,
            },
            "ckd_polarized": {
                "spectral_mode": SpectralMode.CKD,
                "mi_backend": MitsubaBackend.SCALAR,
                "mi_color_mode": MitsubaColorMode.MONO,
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
        return (
            self.mi_backend
            | self.mi_color_mode
            | self.mi_polarized
            | self.mi_double_precision
        )

    @property
    def flags(self) -> ModeFlag:
        return self.spectral_mode | self.mi_flags

    @property
    def spectral_coord_label(self) -> str:
        """
        Spectral coordinate label.
        """
        return _SPECTRAL_COORD_LABELS[self.spectral_mode]

    @property
    def mi_variant(self):
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
        spectral_mode : :.ModeFlag or str, optional
            Spectral mode to check. If unset, the check is skipped.

        mi_backend : :.ModeFlag or str, optional
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
        return self.mi_polarized is True

    @staticmethod
    def new(mode_id: str) -> Mode:
        """
        Create a :class:`Mode` instance given its identifier. Available modes are:

        * ``mono_single``: Monochromatic, single-precision
        * ``mono_double``: Monochromatic, double-precision
        * ``mono``: Alias to ``mono_double``
        * ``ckd_single``: CKD, single-precision
        * ``ckd_double``: CKD, double-precision
        * ``ckd``: Alias to ``ckd_double``

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
        "ckd": _mode_registry()["ckd_double"],
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

    def register(self, mode_flags: ModeFlag | str) -> None:
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
        :meth:`.Mode.flags`, :class:`.ModeFlag`
        """
        if mode_flags is None:
            mode_flags = _active_mode.flags

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


def mode() -> Mode | None:
    """
    Get current operational mode.

    Returns
    -------
    .Mode or None
        Current operational mode.
    """
    return _active_mode


def modes(filter: t.Callable[[Mode], bool] | None = None) -> dict[str, Mode]:
    """
    Get list of registered operational modes.

    Parameters
    ----------
    filter : callable, optional
        A callable used to filter the returned modes. Operates on a
        :class:`.Mode` instance.

    Returns
    -------
    modes: dict[str, .Mode]
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

    return {k: v for k, v in _mode_registry().items() if filter(v)}


def set_mode(mode_id: str):
    """
    Set Eradiate's operational mode.

    This function sets and configures Eradiate's operational mode. Eradiate's
    modes map to Mitsuba's variants and are used to make contextual decisions
    when relevant during the translation of a scene to its kernel format.

    .. admonition:: Valid mode IDs
       :class: info

       * ``mono`` (monochromatic mode, single precision)
       * ``mono_double`` (monochromatic mode, double-precision)
       * ``ckd`` (CKD mode, single precision)
       * ``ckd_double`` (CKD mode, double-precision)
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
    .UnsupportedModeError
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
    .UnsupportedModeError
        Current mode has the requested flags.
    """
    if mode() is None or mode().check(**kwargs):
        raise UnsupportedModeError
