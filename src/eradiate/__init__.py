"""The Eradiate radiative transfer simulation software package."""


from ._version import _version

__version__ = _version  #: Eradiate version string.

# -- Workaround: pre-import modules clashing with Mitsuba's import code --------

__import__("astropy.coordinates")
__import__("xarray")

# -- Global configuration ------------------------------------------------------

from . import _config  # isort: skip

#: Global configuration data structure.
#: Alias to :data:`eradiate._config.config`.
#: See also: :class:`eradiate._config.EradiateConfig`.
config = _config.config

# -- Path resolver -------------------------------------------------------------

# fmt: off
from ._presolver import PathResolver  # isort: skip

#: Global path resolver.
#: See also: :class:`.PathResolver`.
path_resolver = PathResolver()

del PathResolver
# fmt: on

# -- Unit management facilities ------------------------------------------------

from . import units  # isort: skip

#: Global unit registry.
#: Alias to :data:`eradiate.units.unit_registry`.
#: See also: :class:`pint.UnitRegistry`.
unit_registry = units.unit_registry

#: Configuration unit context.
#: Alias to :data:`eradiate.units.unit_context_config`.
#: See also: :class:`pinttr.UnitContext`.
unit_context_config = units.unit_context_config

#: Kernel unit context.
#: Alias to :data:`eradiate.units.unit_context_kernel`.
#: See also: :class:`pinttr.UnitContext`.
unit_context_kernel = units.unit_context_kernel

# -- Operational mode definition -----------------------------------------------

from . import _mode

#: Get the current operational mode.
#: Alias to :func:`eradiate._mode.mode`.
mode = _mode.mode

#: Get a list of registered operational modes.
#: Alias to :func:`eradiate._mode.modes`.
modes = _mode.modes

#: Set Eradiate's operational mode.
#: Alias to :func:`eradiate._mode.set_mode`.
set_mode = _mode.set_mode

#: Raise if the current mode doesn't has specific flags.
#: Alias to :func:`eradiate._mode.supported_mode`.
supported_mode = _mode.supported_mode

#: Raise if the current mode has specific flags.
#: Alias to :func:`eradiate._mode.unsupported_mode`.
unsupported_mode = _mode.unsupported_mode

# ------------------------------------------------------------------------------

from . import (
    ckd,
    contexts,
    converters,
    data,
    experiments,
    kernel,
    notebook,
    pipelines,
    plot,
    rng,
    scenes,
    validators,
    xarray,
)

__all__ = [
    "__version__",
    "ckd",
    "contexts",
    "converters",
    "data",
    "experiments",
    "mode",
    "modes",
    "notebook",
    "path_resolver",
    "pipelines",
    "plot",
    "rng",
    "scenes",
    "set_mode",
    "unit_context_config",
    "unit_context_kernel",
    "unit_registry",
    "units",
    "validators",
    "xarray",
]
