"""Eradiate's computational kernel, based on the Mitsuba 2 rendering system."""

import sys
from importlib import import_module as _import

sys.modules["eradiate.kernel"] = _import("mitsuba")

# Register Mitsuba's C++ modules and submodules
for name in ["core", "render", "core.xml", "core.warp", "core.math",
             "core.spline", "render.mueller"]:
    sys.modules["eradiate.kernel." + name] = _import("mitsuba." + name)


# Cleanup
del sys
del _import, name
