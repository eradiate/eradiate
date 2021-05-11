import enum
from typing import Optional

import attr
import mitsuba

from eradiate._attrs import documented, parse_docs


class ModeSpectrum(enum.Enum):
    """
    An enumeration defining known spectrum representations.
    """

    MONO = "mono"
    # POLY = "poly"
    # SPECTRAL = "spectral"
    # CKD = "ckd"


class ModePrecision(enum.Enum):
    """
    An enumeration defining known kernel precision.
    """

    SINGLE = "single"
    DOUBLE = "double"


# Map associating a mode ID string to the corresponding class
# (aliased in public API section)
_mode_registry = {
    "mono": {
        "spectrum": "mono",
        "precision": "single",
        "kernel_variant": "scalar_mono",
        "spectral_coord_label": "w",
    },
    "mono_double": {
        "spectrum": "mono",
        "precision": "double",
        "kernel_variant": "scalar_mono_double",
        "spectral_coord_label": "w",
    },
}


@parse_docs
@attr.s(frozen=True)
class Mode:
    """
    Data structure describing Eradiate's operational mode and associated ancillary
    data.

    .. important:: Instances are immutable.
    """

    id: str = documented(
        attr.ib(converter=attr.converters.optional(str)),
        doc="Mode identifier.",
        type="str",
    )

    precision: ModePrecision = documented(
        attr.ib(converter=attr.converters.optional(ModePrecision)),
        doc="Mode precision flag.",
        type=":class:`.ModePrecision`",
    )

    spectrum: ModeSpectrum = documented(
        attr.ib(converter=attr.converters.optional(ModeSpectrum)),
        doc="Mode spectrum flag.",
        type=":class:`.ModeSpectrum`",
    )

    kernel_variant: str = documented(
        attr.ib(converter=attr.converters.optional(str)),
        doc="Mode kernel variant.",
        type="str",
    )

    spectral_coord_label: str = documented(
        attr.ib(converter=attr.converters.optional(str)),
        doc="Mode spectral coordinate label.",
        type="str",
    )

    @staticmethod
    def new(mode_id):
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

    def is_monochromatic(self):
        """
        Return ``True`` if active mode is monochromatic.
        """
        return self.spectrum is ModeSpectrum.MONO

    def is_single_precision(self):
        """
        Return ``True`` if active mode is single-precision.
        """
        return self.precision is ModePrecision.SINGLE

    def is_double_precision(self):
        """
        Return ``True`` if active mode is double-precision.
        """
        return self.precision is ModePrecision.DOUBLE


# Eradiate's operational mode configuration
_current_mode = None


# -- Public API ----------------------------------------------------------------


def mode() -> Optional[Mode]:
    """
    Get current operational mode.

    Returns → :class:`Mode`
        Current operational mode.
    """
    return _current_mode


def modes():
    """
    Get list of registered operational modes.

    Returns → dict[str, Mode]
        List of registered operational modes
    """
    return _mode_registry


def set_mode(mode_id):
    """
    Set Eradiate's operational mode.

    This function sets and configures Eradiate's operational mode. Eradiate's
    modes map to Mitsuba's variants and are used to make contextual decisions
    when relevant during the translation of a scene to its kernel format.

    Parameter ``mode_id`` (str):
        Mode to be selected (see list below).

    .. rubric:: Available modes

    * ``mono`` (monochromatic mode, single precision)
    * ``mono_double`` (monochromatic mode, double-precision)
    """
    global _current_mode

    if mode_id in _mode_registry.keys():
        mode = Mode.new(mode_id)
        mitsuba.set_variant(mode.kernel_variant)
    else:
        mode = None

    _current_mode = mode
