from importlib.metadata import PackageNotFoundError, version

# Internal constants
_EXPECTED_MITSUBA_VERSION = "3.2.1"
_EXPECTED_MITSUBA_PATCH_VERSION = "0.1.1rc1"

# Exported constants
ERADIATE_MITSUBA_PACKAGE_VERSION = None
ERADIATE_MITSUBA_PACKAGE = False
ERADIATE_KERNEL_VERSION = None
ERADIATE_KERNEL_PATCH_VERSION = None

# Check if Dr.Jit and Mitsuba can be imported successfully
try:
    __import__("drjit")
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Could not import module 'drjit'; did you build the kernel and add "
        "it to your $PYTHONPATH?"
    ) from e

try:
    _mi = __import__("mitsuba")
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Could not import module 'mitsuba'; did you build the kernel and add "
        "it to your $PYTHONPATH?"
    ) from e

# Check if the eradiate-mitsuba package is installed
try:
    ERADIATE_MITSUBA_PACKAGE_VERSION = version("eradiate-mitsuba")
    ERADIATE_MITSUBA_PACKAGE = True
except PackageNotFoundError as pkg_error:
    pass

# Retrieve the kernel and kernel patch versions
ERADIATE_KERNEL_VERSION = _mi.scalar_mono.MI_VERSION
if hasattr(_mi.scalar_rgb, "ERD_MI_VERSION"):
    ERADIATE_KERNEL_PATCH_VERSION = _mi.scalar_rgb.ERD_MI_VERSION


import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
