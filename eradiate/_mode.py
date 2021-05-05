import enum
from typing import Optional

import attr
import mitsuba


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
    },
    "mono_double": {
        "spectrum": "mono",
        "precision": "double",
        "kernel_variant": "scalar_mono_double",
    },
}


@attr.s(frozen=True)
class Mode:
    id = attr.ib()
    precision = attr.ib(converter=attr.converters.optional(ModePrecision))
    spectrum = attr.ib(converter=attr.converters.optional(ModeSpectrum))
    kernel_variant = attr.ib()

    @staticmethod
    def new(mode_id):
        try:
            mode_kwargs = _mode_registry[mode_id]
        except KeyError:
            raise ValueError(f"unknown mode '{mode_id}'")
        return Mode(id=mode_id, **mode_kwargs)

    def is_monochromatic(self):
        return self.spectrum is ModeSpectrum.MONO

    def is_single_precision(self):
        return self.precision is ModePrecision.SINGLE

    def is_double_precision(self):
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
