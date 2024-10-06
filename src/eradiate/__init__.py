"""The Eradiate radiative transfer simulation software package."""

from ._version import _version
from .kernel import check_kernel

check_kernel()
__version__ = _version  #: Eradiate version string.

# -- Lazy imports ------------------------------------------------------

import lazy_loader  # noqa: E402

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
