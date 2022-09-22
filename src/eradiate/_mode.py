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


class ModeFlags(enum.Flag):
    """
    Flags defining Eradiate mode features.
    """

    # -- Mode feature flags ----------------------------------------------------

    # Eradiate
    ERT_MONO = enum.auto()  #: Mode performs monochromatic simulations
    ERT_CKD = enum.auto()  #: Mode performs correlated-k spectral simulation
    # ERT_RGB = enum.auto()  #: Mode performs renders RGB images

    # Mitsuba
    MI_SCALAR = enum.auto()  #: Mode maps to a scalar Mitsuba variant
    # MI_LLVM = enum.auto()  #: Mode maps to a LLVM Mitsuba variant
    # MI_AD = enum.auto()  #: Mode maps to an autodiff Mitsuba variant
    MI_MONO = enum.auto()  #: Mode maps to a monochromatic Mitsuba variant
    # MI_RGB = enum.auto()  #: Mode maps to an RGB Mitsuba variant
    # MI_SPECTRAL = enum.auto()  #: Mode maps to a spectral Mitsuba variant
    # MI_UNPOLARIZED = enum.auto() #: Mode maps to an unpolarised Mitsuba variant
    # MI_POLARIZED = enum.auto() #: Mode maps to a polarised Mitsuba variant
    MI_SINGLE = enum.auto()  #: Mode maps to a single-precision Mitsuba variant
    MI_DOUBLE = enum.auto()  #: Mode maps to a double-precision Mitsuba variant

    # -- Mode definition flags -------------------------------------------------

    MONO_SINGLE = (
        ERT_MONO | MI_SCALAR | MI_MONO | MI_SINGLE
    )  #: Monochromatic mode, single precision
    MONO_DOUBLE = (
        ERT_MONO | MI_SCALAR | MI_MONO | MI_DOUBLE
    )  #: Monochromatic mode, double precision
    CKD_SINGLE = (
        ERT_CKD | MI_SCALAR | MI_MONO | MI_SINGLE
    )  #: CKD mode, single precision
    CKD_DOUBLE = (
        ERT_CKD | MI_SCALAR | MI_MONO | MI_DOUBLE
    )  #: CKD mode, double precision

    # -- Other convenience aliases ---------------------------------------------

    ANY_MONO = ERT_MONO  #: Any monochromatic mode
    ANY_CKD = ERT_CKD  #: Any CKD mode
    ANY_SCALAR = MI_SCALAR  #: Any scalar mode
    ANY_SINGLE = MI_SINGLE  #: Any single-precision mode
    ANY_DOUBLE = MI_DOUBLE  #: Any double-precision mode


# ------------------------------------------------------------------------------
#                              Mode definitions
# ------------------------------------------------------------------------------

# Map associating a mode ID string to the corresponding class
# (aliased in public API section)
_mode_registry = {
    "mono_single": {
        "flags": ModeFlags.MONO_SINGLE,
        "spectral_coord_label": "w",
    },
    "mono_double": {
        "flags": ModeFlags.MONO_DOUBLE,
        "spectral_coord_label": "w",
    },
    "mono": {  # Alias to mono_double
        "flags": ModeFlags.MONO_DOUBLE,
        "spectral_coord_label": "w",
    },
    "ckd_single": {
        "flags": ModeFlags.CKD_SINGLE,
        "spectral_coord_label": "bd",
    },
    "ckd_double": {
        "flags": ModeFlags.CKD_DOUBLE,
        "spectral_coord_label": "bd",
    },
    "ckd": {  # Alias to ckd_double
        "flags": ModeFlags.CKD_DOUBLE,
        "spectral_coord_label": "bd",
    },
}


@parse_docs
@attrs.frozen
class Mode:
    """
    Data structure describing Eradiate's operational mode and associated
    ancillary data.

    .. important:: Instances are immutable.
    """

    id: str = documented(
        attrs.field(converter=attrs.converters.optional(str)),
        doc="Mode identifier.",
        type="str",
    )

    flags: ModeFlags = documented(
        attrs.field(converter=ModeFlags),
        doc="Mode flags.",
        type=":class:`.ModeFlags`",
    )

    spectral_coord_label: str = documented(
        attrs.field(converter=attrs.converters.optional(str)),
        doc="Mode spectral coordinate label.",
        type="str",
    )

    @property
    def kernel_variant(self) -> str:
        """Mode kernel variant."""
        components = []

        # Backend selection
        if self.flags & ModeFlags.MI_SCALAR:
            components.append("scalar")

        # Todo: Autodiff mode selection

        # Spectral mode selection
        if self.flags & ModeFlags.MI_MONO:
            components.append("mono")

        # Todo: Polarisation mode selection

        # Precision mode selection
        if self.flags & ModeFlags.MI_DOUBLE:
            components.append("double")

        return "_".join(components)

    def has_flags(self, flags: t.Union[ModeFlags, str]) -> bool:
        """
        Check if the currently active mode has the passed flags.

        Parameters
        ----------
        flags : .ModeFlags or str
            Flags to check for. If a string is passed, conversion to a
            :class:`.ModeFlags` instance will be attempted.

        Returns
        -------
        bool
            ``True`` if current mode has the passed flags, ``False`` otherwise.
        """
        if isinstance(flags, str):
            flags = ModeFlags[flags.upper()]
        return bool(self.flags & flags)

    @property
    def is_mono(self):
        return bool(self.flags & ModeFlags.ANY_MONO)

    @property
    def is_ckd(self):
        return bool(self.flags & ModeFlags.ANY_CKD)

    @property
    def is_single_precision(self):
        return bool(self.flags & ModeFlags.ANY_SINGLE)

    @property
    def is_double_precision(self):
        return bool(self.flags & ModeFlags.ANY_DOUBLE)

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
_active_mode: t.Optional[Mode] = None


# ------------------------------------------------------------------------------
#                                Public API
# ------------------------------------------------------------------------------


def mode() -> t.Optional[Mode]:
    """
    Get current operational mode.

    Returns
    -------
    .Mode or None
        Current operational mode.
    """
    return _active_mode


def modes() -> t.Dict:
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
        mitsuba.set_variant(mode.kernel_variant)
    elif mode_id.lower() == "none":
        mode = None
    else:
        raise ValueError(f"unknown mode '{mode_id}'")

    _active_mode = mode


def supported_mode(flags):
    """
    Check whether the current mode has specific flags. If not, raise.

    Parameters
    ----------
    flags : .ModeFlags
        Flags the current mode is expected to have.

    Raises
    ------
    .UnsupportedModeError
        Current mode does not have the requested flags.
    """
    if mode() is None or not mode().has_flags(flags):
        raise UnsupportedModeError


def unsupported_mode(flags):
    """
    Check whether the current mode has specific flags. If so, raise.

    Parameters
    ----------
    flags : .ModeFlags
        Flags the current mode is expected not to have.

    Raises
    ------
    .UnsupportedModeError
        Current mode has the requested flags.
    """
    if mode() is None or mode().has_flags(flags):
        raise UnsupportedModeError
