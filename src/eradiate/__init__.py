"""The Eradiate radiative transfer simulation software package."""

import importlib
import os

from ._version import _version

__version__ = _version  #: Eradiate version string.

# -- Lazy imports ------------------------------------------------------

import lazy_loader  # noqa: E402

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader

if not os.environ.get("EAGER_IMPORT"):
    # This performs kernel checks. If eager imports are activated, this import
    # will occur automatically, and it is therefore not necessary to repeat it.
    importlib.import_module("eradiate.kernel._versions")
