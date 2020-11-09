"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

# -- Operational mode definition -----------------------------------------------

import enum

import attr

from .util.attrs import attrib_quantity, unit_enabled
from .util.units import config_default_units, ureg


class ModeSpectrum(enum.Enum):
    """An enumeration defining known kernel spectrum representations."""

    MONO = "mono"
    # POLY = "poly"
    # SPECTRAL = "spectral"
    # CKD = "ckd"


class ModePrecision(enum.Enum):
    """An enumeration defining known kernel precision."""

    SINGLE = "single"
    DOUBLE = "double"


# Map associating a mode ID string to the corresponding class
# (aliased in public API section)
mode_registry = {}


def register_mode(mode_id, spectrum="mono", precision="single"):
    # This decorator is meant to be added to added to an Mode child class.
    # It adds it to mode_registry.

    spectrum = ModeSpectrum(spectrum)
    precision = ModePrecision(precision)

    def decorator(cls):
        mode_registry[mode_id] = cls
        cls.id = mode_id
        cls.precision = precision
        cls.spectrum = spectrum
        return cls

    return decorator


@unit_enabled
@attr.s(frozen=True)
class Mode:
    # Parent class for all operational modes
    # All derived classes will be frozen
    # Side-effect: they will also implement from_dict()
    id = None
    precision = None
    spectrum = None

    def is_monochromatic(self):
        return self.spectrum is ModeSpectrum.MONO

    def is_double_precision(self):
        return self.precision is ModePrecision.DOUBLE

    @staticmethod
    def new(mode_id, **kwargs):
        try:
            mode_cls = mode_registry[mode_id]
        except KeyError:
            raise ValueError(f"unknown mode '{mode_id}'")
        return mode_cls(**kwargs)


@register_mode("none")
@attr.s
class ModeNone(Mode):
    # Default mode, defined to improve error messages
    pass


@register_mode("mono", spectrum="mono", precision="single")
@attr.s
class ModeMono(Mode):
    # Monochromatic mode, single precision
    wavelength = attrib_quantity(
        default=ureg.Quantity(550., ureg.nm),
        units_compatible=config_default_units.generator("wavelength"),
        on_setattr=None
    )

    def __attrs_post_init__(self):
        import eradiate.kernel
        eradiate.kernel.set_variant("scalar_mono")


@register_mode("mono_double", spectrum="mono", precision="double")
@attr.s
class ModeMonoDouble(Mode):
    # Monochromatic mode, double precision
    wavelength = attrib_quantity(
        default=ureg.Quantity(550., ureg.nm),
        units_compatible=config_default_units.generator("wavelength"),
        on_setattr=None
    )

    def __attrs_post_init__(self):
        import eradiate.kernel
        eradiate.kernel.set_variant("scalar_mono_double")


# -- Public API ----------------------------------------------------------------

mode = ModeNone()
"""Eradiate's operational mode configuration.

.. seealso::

   :func:`set_mode`.
"""

#: Registered mode list
modes = mode_registry


def set_mode(mode_id, **kwargs):
    """Set Eradiate's mode of operation.

    This function sets and configures Eradiate's mode of operation. In addition,
    it invokes the :func:`~mitsuba.set_variant` kernel function to select the
    kernel variant corresponding to the selected mode.

    The main argument ``mode_id`` defines which mode is selected. Then,
    keyword arguments are used to pass additional configuration details for the
    selected mode. The mode configuration is critical since many code components
    (*e.g.* spectrum-related components) adapt their behaviour based on the
    selected mode.

    Parameter ``mode_id`` (str):
        Mode to be selected (see list below).

    .. rubric:: Available modes and corresponding keyword arguments

    ``mono`` (monochromatic mode, single precision)
        ``wavelength`` (float):
            Wavelength selected for monochromatic operation. Default: 550 nm.

            Unit-enabled field (default: cdu[wavelength]).

    ``mono_double`` (monochromatic mode, double-precision)
        ``wavelength`` (float):
            Wavelength selected for monochromatic operation. Default: 550 nm.

            Unit-enabled field (default: cdu[wavelength]).
    """
    global mode
    mode = Mode.new(mode_id, **kwargs)


# -- Required imports ----------------------------------------------------------

from .util.xarray import EradiateDataArrayAccessor, EradiateDatasetAccessor

# -- Cleanup -------------------------------------------------------------------

del enum
del attr
del attrib_quantity, unit_enabled
del config_default_units, ureg
del EradiateDataArrayAccessor, EradiateDatasetAccessor
