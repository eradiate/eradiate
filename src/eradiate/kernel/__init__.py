# Check if Dr.Jit and Mitsuba can be imported successfully
try:
    __import__("drjit")
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Could not import module 'drjit'; did you build the kernel and add "
        "it to your $PYTHONPATH?"
    ) from e

try:
    __import__("mitsuba")
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Could not import module 'mitsuba'; did you build the kernel and add "
        "it to your $PYTHONPATH?"
    ) from e

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
