from __future__ import annotations

import enum
import typing as t

import attrs
import mitsuba

from .attrs import documented, parse_docs
from .exceptions import UnsupportedModeError

# ------------------------------------------------------------------------------
#                                 Mode flags
# ------------------------------------------------------------------------------


class Flag(enum.Flag):
    """
    Small extension to :class:`enum.Flag` that adds a ``convert()`` class method
    constructor.
    """

    @classmethod
    def convert(cls, value: t.Any) -> Flag:
        """
        Try to convert a value to a flag. Strings are capitalized and converted
        to the corresponding enum member.
        """
        if isinstance(value, str):
            return cls[value.upper()]
        else:
            return cls(value)


class SpectralMode(Flag):
    """
    Spectral dimension handling flags.
    """

    MONO = enum.auto()  #: Monochromatic (line-by-line) mode
    CKD = enum.auto()  #: Correlated-k distribution mode


_SPECTRAL_COORD_LABELS = {SpectralMode.MONO: "w", SpectralMode.CKD: "bd"}


class MitsubaBackend(Flag):
    """
    Mitsuba backend flags.
    """

    SCALAR = enum.auto()  #: Scalar backend
    LLVM = enum.auto()  #: LLVM backend


class MitsubaColorMode(Flag):
    """
    Mitsuba color mode flags.
    """

    MONO = enum.auto()  #: Monochromatic mode


# ------------------------------------------------------------------------------
#                              Mode definitions
# ------------------------------------------------------------------------------

# Map associating a mode ID string to the corresponding class
# (aliased in public API section)
_mode_registry: dict[str, dict] = {
    "mono_single": {
        "spectral_mode": SpectralMode.MONO,
        "mi_backend": MitsubaBackend.SCALAR,
        "mi_color_mode": MitsubaColorMode.MONO,
        "mi_double_precision": False,
        "mi_polarized": False,
    },
    "mono_double": {
        "spectral_mode": SpectralMode.MONO,
        "mi_backend": MitsubaBackend.SCALAR,
        "mi_color_mode": MitsubaColorMode.MONO,
        "mi_double_precision": True,
        "mi_polarized": False,
    },
    "ckd_single": {
        "spectral_mode": SpectralMode.CKD,
        "mi_backend": MitsubaBackend.SCALAR,
        "mi_color_mode": MitsubaColorMode.MONO,
        "mi_double_precision": False,
        "mi_polarized": False,
    },
    "ckd_double": {
        "spectral_mode": SpectralMode.CKD,
        "mi_backend": MitsubaBackend.SCALAR,
        "mi_color_mode": MitsubaColorMode.MONO,
        "mi_double_precision": True,
        "mi_polarized": False,
    },
}

# Aliases
_mode_registry.update(
    {
        "mono": _mode_registry["mono_double"].copy(),
        "ckd": _mode_registry["ckd_double"].copy(),
    }
)


@parse_docs
@attrs.frozen
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

    spectral_mode: SpectralMode = documented(
        attrs.field(converter=SpectralMode.convert),
        doc="Spectral dimension handling.",
        type=":class:`.SpectralMode`",
        init_type=":class:`.SpectralMode` or str",
    )

    mi_backend: MitsubaBackend = documented(
        attrs.field(converter=MitsubaBackend.convert),
        doc="Mitsuba computational backend.",
        type=":class:`.MitsubaBackend`",
        init_type=":class:`.MitsubaBackend` or str",
    )

    mi_color_mode: MitsubaColorMode = documented(
        attrs.field(converter=MitsubaColorMode.convert),
        doc="Mitsuba color mode.",
        type=".MitsubaColorMode",
        init_type=":class:`.MitsubaColorMode` or str",
    )

    mi_polarized: bool = documented(
        attrs.field(default=False, converter=bool),
        doc="Mitsuba polarized mode.",
        type="bool",
        default="False",
    )

    mi_double_precision: bool = documented(
        attrs.field(default=True, converter=bool),
        doc="Mitsuba double precision.",
        type="bool",
        default="True",
    )

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

        result = [self.mi_backend.name.lower(), self.mi_color_mode.name.lower()]
        if self.mi_polarized:
            result.append("polarized")
        if self.mi_double_precision:
            result.append("double")
        return "_".join(result)

    def check(
        self,
        spectral_mode: None | SpectralMode | str = None,
        mi_backend: None | MitsubaBackend | str = None,
        mi_color_mode: None | MitsubaColorMode | str = None,
        mi_polarized: bool | None = None,
        mi_double_precision: bool | None = None,
    ) -> bool:
        """
        Check if the currently active mode has the passed flags.

        Parameters
        ----------
        spectral_mode : :class:`.SpectralMode` or str, optional
            Spectral mode to check. If unset, the check is skipped.

        mi_backend : :class:`.MitsubaBackend` or str, optional
            Mitsuba backend to check. If unset, the check is skipped.

        mi_color_mode : :class:`.MitsubaColorMode` or str, optional
            Mitsuba color mode to check. If unset, the check is skipped.

        mi_polarized : bool, optional
            Mitsuba polarized mode to check. If unset, the check is skipped.

        mi_double_precision : bool, optional
            Mitsuba double precision mode to check. If unset, the check is skipped.

        Returns
        -------
        bool
            ``True`` if current mode has the passed flags, ``False`` otherwise.
        """
        outcome = True

        if spectral_mode is not None:
            outcome &= bool(self.spectral_mode & SpectralMode.convert(spectral_mode))

        if mi_backend is not None:
            outcome &= bool(self.mi_backend & MitsubaBackend.convert(mi_backend))

        if mi_color_mode is not None:
            outcome &= bool(
                self.mi_color_mode & MitsubaColorMode.convert(mi_color_mode)
            )

        if mi_polarized is not None:
            outcome &= self.mi_polarized == mi_polarized

        if mi_double_precision is not None:
            outcome &= self.mi_double_precision == mi_double_precision

        return outcome

    @property
    def is_mono(self) -> bool:
        return self.spectral_mode is SpectralMode.MONO

    @property
    def is_ckd(self) -> bool:
        return self.spectral_mode is SpectralMode.CKD

    @property
    def is_single_precision(self) -> bool:
        return self.mi_double_precision is False

    @property
    def is_double_precision(self) -> bool:
        return self.mi_double_precision is True

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
        :class:`Mode`
            Created :class:`Mode` instance.
        """
        try:
            mode_kwargs = _mode_registry[mode_id]
        except KeyError:
            raise ValueError(f"unknown mode '{mode_id}'")
        return Mode(id=mode_id, **mode_kwargs)


# Eradiate's operational mode configuration
_active_mode: Mode | None = None


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


def modes() -> dict:
    """
    Get list of registered operational modes.

    Returns
    -------
    dict
        List of registered operational modes
    """
    return _mode_registry


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

    if mode_id in _mode_registry:
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
