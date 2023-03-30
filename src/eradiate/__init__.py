"""The Eradiate radiative transfer simulation software package."""

from ._version import _version

__version__ = _version  #: Eradiate version string.

# -- Lazy imports ------------------------------------------------------

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
