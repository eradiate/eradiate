# Import accessors temporarily to set them up
from . import _accessors  # isort: skip

del _accessors


# -- Lazy imports ------------------------------------------------------

from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules=["interp"],
)

del lazy_loader
