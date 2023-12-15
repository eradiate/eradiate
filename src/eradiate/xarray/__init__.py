# Import accessors temporarily to set them up
from . import _accessors  # isort: skip

del _accessors

# Lazy imports
import lazy_loader  # noqa: E402

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
