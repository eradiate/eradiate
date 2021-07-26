from __future__ import annotations

import enum
from typing import Dict, Optional, Union

import attr
import mitsuba

from .attrs import documented, parse_docs

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
    # ERT_CKD = enum.auto()  #: Mode performs correlated-k spectral simulation
    # ERT_RGB = enum.auto()  #: Mode performs renders RGB images

    # Mitsuba
    MTS_SCALAR = enum.auto()  #: Mode maps to a scalar Mitsuba variant
    # MTS_LLVM = enum.auto()  #: Mode maps to a LLVM Mitsuba variant
    MTS_MONO = enum.auto()  #: Mode maps to a monochromatic Mitsuba variant
    # MTS_RGB = enum.auto()  #: Mode maps to an RGB Mitsuba variant
    # MTS_POLARIZED = enum.auto() #: Mode maps to a polarised RGB Mitsuba variant
    MTS_DOUBLE = enum.auto()  #: Mode maps to a double-precision Mitsuba variant

    # -- Mode definition flags -------------------------------------------------

    MONO = ERT_MONO | MTS_SCALAR | MTS_MONO
    MONO_DOUBLE = ERT_MONO | MTS_SCALAR | MTS_MONO | MTS_DOUBLE
    # CKD = ERT_CKD | MTS_SCALAR | MTS_MONO
    # CKD_DOUBLE = ERT_CKD | MTS_SCALAR | MTS_MONO | MTS_DOUBLE

    # -- Other convenience aliases ---------------------------------------------

    ANY_MONO = ERT_MONO
    ANY_SCALAR = MTS_SCALAR
    ANY_DOUBLE = MTS_DOUBLE


# ------------------------------------------------------------------------------
#                              Mode definitions
# ------------------------------------------------------------------------------

# Map associating a mode ID string to the corresponding class
# (aliased in public API section)
_mode_registry = {
    "mono": {
        "flags": ModeFlags.MONO,
        "spectral_coord_label": "w",
    },
    "mono_double": {
        "flags": ModeFlags.MONO_DOUBLE,
        "spectral_coord_label": "w",
    },
}


@parse_docs
@attr.s(frozen=True, slots=True)
class Mode:
    """
    Data structure describing Eradiate's operational mode and associated
    ancillary data.

    .. important:: Instances are immutable.
    """

    id: str = documented(
        attr.ib(converter=attr.converters.optional(str)),
        doc="Mode identifier.",
        type="str",
    )

    flags: ModeFlags = documented(
        attr.ib(converter=ModeFlags),
        doc="Mode flags.",
        type=":class:`.ModeFlags`",
    )

    spectral_coord_label: str = documented(
        attr.ib(converter=attr.converters.optional(str)),
        doc="Mode spectral coordinate label.",
        type="str",
    )

    @property
    def kernel_variant(self) -> str:
        """Mode kernel variant."""
        components = []

        # Backend selection
        if self.flags & ModeFlags.MTS_SCALAR:
            components.append("scalar")

        # Todo: Autodiff mode selection

        # Spectral mode selection
        if self.flags & ModeFlags.MTS_MONO:
            components.append("mono")

        # Todo: Polarisation mode selection

        # Precision mode selection
        if self.flags & ModeFlags.MTS_DOUBLE:
            components.append("double")

        return "_".join(components)

    def has_flags(self, flags: Union[ModeFlags, str]):
        """
        Check if the currently active mode has the passed flags.
        """
        if isinstance(flags, str):
            flags = ModeFlags[flags.upper()]
        return self.flags & flags

    @staticmethod
    def new(mode_id) -> Mode:
        """
        Create a :class:`Mode` instance given its identifier. Available modes are:

        * ``mono``: Monochromatic, single-precision
        * ``mono_double``: Monochromatic, double-precision
        """
        try:
            mode_kwargs = _mode_registry[mode_id]
        except KeyError:
            raise ValueError(f"unknown mode '{mode_id}'")
        return Mode(id=mode_id, **mode_kwargs)


# Eradiate's operational mode configuration
_current_mode: Optional[Mode] = None


# ------------------------------------------------------------------------------
#                                Public API
# ------------------------------------------------------------------------------


def mode() -> Optional[Mode]:
    """
    Get current operational mode.

    Returns → :class:`.Mode` or None
        Current operational mode.
    """
    return _current_mode


def modes() -> Dict:
    """
    Get list of registered operational modes.

    Returns → dict
        List of registered operational modes
    """
    return _mode_registry


def set_mode(mode_id: str):
    """
    Set Eradiate's operational mode.

    This function sets and configures Eradiate's operational mode. Eradiate's
    modes map to Mitsuba's variants and are used to make contextual decisions
    when relevant during the translation of a scene to its kernel format.

    Parameter ``mode_id`` (str):
        Mode to be selected (see list below).

    Raises → ValueError:
        ``mode_id`` does not match any of the known mode identifiers.

    .. rubric:: Valid mode IDs

    * ``mono`` (monochromatic mode, single precision)
    * ``mono_double`` (monochromatic mode, double-precision)
    * ``none`` (no mode selected)
    """
    global _current_mode

    if mode_id in _mode_registry:
        mode = Mode.new(mode_id)
        mitsuba.set_variant(mode.kernel_variant)
    elif mode_id.lower() == "none":
        mode = None
    else:
        raise ValueError(f"unknown mode '{mode_id}'")

    _current_mode = mode
