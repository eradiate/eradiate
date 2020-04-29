import sys
from importlib import import_module as _import

# Register Mitsuba's modules and submodules
for name in ["core", "render", "core.xml", "core.warp", "core.math",
             "core.spline", "render.mueller"]:
    sys.modules["eradiate.kernel." + name] = _import("mitsuba." + name)

# Import other top-level Mitsuba functions
from mitsuba import variant, variants, set_variant

# Cleanup
del sys
del name
