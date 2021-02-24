"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

# -- Operational mode definition -----------------------------------------------

from ._mode import mode, set_mode, modes

# -- Unit management facilities ------------------------------------------------

from ._units import unit_registry, unit_context_config, unit_context_kernel

# -- Path resolver -------------------------------------------------------------

from ._presolver import PathResolver
path_resolver = PathResolver()
del PathResolver

# ------------------------------------------------------------------------------

from . import scenes, solvers
from . import converters, validators, xarray

__all__ = [
    "__version__",
    "mode",
    "modes",
    "set_mode",
    "path_resolver",
    "unit_registry",
    "unit_context_config",
    "unit_context_kernel",
    "scenes",
    "solvers",
    "converters",
    "validators",
    "xarray",
]
