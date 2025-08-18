from __future__ import annotations

from enum import IntEnum


class FilterType(IntEnum):
    """
    Filter type for scene elements.

    This enum controls whether interactions with scene elements (BSDFs, Shapes, etc.)
    should be included or ignored in sensor measurements.
    """

    INCLUDE = 0  # Include this element in measurements (default)
    IGNORE = 1  # Ignore this element in measurements
