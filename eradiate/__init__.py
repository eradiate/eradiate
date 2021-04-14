"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

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

from . import converters, kernel, scenes, solvers, validators, xarray

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
