"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

# -- Workaround: pre-import modules clashing with Mitsuba's import code --------

__import__("xarray")

# -- Global configuration ------------------------------------------------------

from ._config import config  # isort: skip

# -- Path resolver -------------------------------------------------------------

# fmt: off
from ._presolver import PathResolver  # isort: skip
path_resolver = PathResolver()
del PathResolver
# fmt: on

# -- Unit management facilities ------------------------------------------------

from .units import (  # isort: skip
    unit_context_config,
    unit_context_kernel,
    unit_registry,
)

# -- Operational mode definition -----------------------------------------------

from ._mode import mode, set_mode, modes  # isort: skip

# ------------------------------------------------------------------------------

from . import (
    contexts,
    converters,
    kernel,
    notebook,
    scenes,
    solvers,
    units,
    validators,
    xarray,
)

__all__ = [
    "__version__",
    "contexts",
    "converters",
    "mode",
    "modes",
    "notebook",
    "path_resolver",
    "scenes",
    "set_mode",
    "solvers",
    "unit_context_config",
    "unit_context_kernel",
    "unit_registry",
    "units",
    "validators",
    "xarray",
]
