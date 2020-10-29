"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

import attr

from .util.attrs import attrib_quantity as _attrib_quantity
from .util.attrs import converter_to_units as _converter_to_units
from .util.attrs import unit_enabled as _unit_enabled
from .util.collections import configdict as _configdict
from .util.units import config_default_units as _cdu
from .util.units import ureg as _ureg

mode = None
"""Eradiate's operational mode configuration.

.. seealso::

   :func:`set_mode`.
"""

# Map associating a mode ID string to the corresponding class
_registered_modes = {}


def _register_mode(mode_id):
    # This decorator is meant to be added to added to an _EradiateMode child
    # class. It adds it to _registered_modes.

    def decorator(cls):
        _registered_modes[mode_id] = cls
        cls.id = mode_id
        return cls

    return decorator


@_unit_enabled
@attr.s(frozen=True)
class _EradiateMode:
    # Parent class for all operational modes
    # All derived classes will be frozen
    # Side-effect: they will also implement from_dict()
    id = None

    @staticmethod
    def new(mode_id, **kwargs):
        try:
            mode_cls = _registered_modes[mode_id]
        except KeyError:
            raise ValueError(f"unknown mode '{mode_id}'")
        return mode_cls(**kwargs)


@_register_mode("mono")
@attr.s
class _EradiateModeMono(_EradiateMode):
    # Monochromatic mode
    wavelength = _attrib_quantity(
        default=_ureg.Quantity(550., _ureg.nm),
        units_compatible=_cdu.generator("wavelength"),
        on_setattr=None
    )

    def __attrs_post_init__(self):
        import eradiate.kernel
        eradiate.kernel.set_variant("scalar_mono_double")


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

    ``mono`` (monochromatic mode)
        ``wavelength`` (float):
            Wavelength selected for monochromatic operation. Default: 550 nm.

            Unit-enabled field (default: cdu[wavelength]).
    """
    global mode
    mode = _EradiateMode.new(mode_id, **kwargs)
