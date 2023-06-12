from importlib.metadata import version, PackageNotFoundError
import re, pathlib

# Internal constants
_EXPECTED_MITSUBA_VERSION = "3.2.1"
_EXPECTED_MITSUBA_PATCH_VERSION = "0.0.1"

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

# Parse the kernel and kernel patch versions
try:
    mi_version_regex = re.compile(
                r"^\s*#\s*define\s+MI_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE)
    mi_patch_version_regex = re.compile(
                r"^\s*#\s*define\s+ERD_MI_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE)
    for path in _mi.__path__:
        path = pathlib.Path(path) / "include" / "mitsuba" / "mitsuba.h"
        if path.exists() and path.is_file():
            with path.open("r") as f:
                header = f.read()
                matches = dict(mi_version_regex.findall(header))
                patch_matches = dict(mi_patch_version_regex.findall(header))
                ERADIATE_KERNEL_VERSION = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)
                ERADIATE_KERNEL_PATCH_VERSION = "{MAJOR}.{MINOR}.{PATCH}".format(**patch_matches)
                break
except Exception as e:
    raise ImportError(
        "Could not parse the Eradiate patch version from the Mitsuba kernel header."
    ) from e
if ERADIATE_KERNEL_VERSION is None:
    raise ImportError("Could not find the Mitsuba header file.")

# Check if the kernel version is compatible
if ERADIATE_KERNEL_VERSION != _EXPECTED_MITSUBA_VERSION:
    raise ImportError(
        "Using an incompatible version of Mitsuba. Eradiate requires Mitsuba "
        f"{_EXPECTED_MITSUBA_VERSION}. Found Mitsuba {ERADIATE_KERNEL_VERSION}."
    )
if ERADIATE_KERNEL_PATCH_VERSION != _EXPECTED_MITSUBA_PATCH_VERSION:
    raise ImportError(
        "Using an incompatible patch version of Mitsuba. Eradiate requires a "
        f"Mitsuna kernel version {_EXPECTED_MITSUBA_VERSION}, with the patch "
        f"version {_EXPECTED_MITSUBA_PATCH_VERSION}. Found Mitsuba "
        f"{ERADIATE_KERNEL_VERSION} with the patch version"
        f" {ERADIATE_KERNEL_PATCH_VERSION}."
    )

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
