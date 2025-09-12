from __future__ import annotations

from enum import Flag, IntEnum, auto


class FilterFlags(Flag):
    """
    Flags for different types of scene elements that can be filtered.
    """

    BSDF = auto()
    SHAPE = auto()
    PHASE = auto()


class FilterType(IntEnum):
    """
    Filter type for scene elements.

    This enum controls whether interactions with scene elements (BSDFs, Shapes, etc.)
    should be included or ignored in sensor measurements.
    """

    INCLUDE = 0  # Include this element in measurements (default)
    IGNORE = 1  # Ignore this element in measurements
