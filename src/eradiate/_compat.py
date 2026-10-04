"""
Compatibility shims for older Python and dependency versions.

Each entry states the version it supports; remove it when that version is
dropped.
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["ParamSpec", "TypeAlias", "trapezoid"]

# Python < 3.10
if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeAlias
else:
    from typing_extensions import ParamSpec, TypeAlias

# NumPy < 2.0
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # noqa: NPY201
