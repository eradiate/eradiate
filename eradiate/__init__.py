"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

# -- Workaround: pre-import modules clashing with Mitsuba's import code --------

__import__("xarray")

# -- Operational mode definition -----------------------------------------------

from ._mode import mode, set_mode, modes  # isort: skip

# -- Unit management facilities ------------------------------------------------

from ._units import (
    unit_registry,
    unit_context_config,
    unit_context_kernel,
)  # isort: skip

# -- Path resolver -------------------------------------------------------------

# fmt: off
from ._presolver import PathResolver  # isort: skip
path_resolver = PathResolver()
del PathResolver
# fmt: on

# ------------------------------------------------------------------------------

from . import converters, contexts, kernel, scenes, solvers, validators, xarray

__all__ = [
    "__version__",
    "converters",
    "contexts",
    "mode",
    "modes",
    "path_resolver",
    "scenes",
    "set_mode",
    "solvers",
    "unit_context_config",
    "unit_context_kernel",
    "unit_registry",
    "validators",
    "xarray",
]
