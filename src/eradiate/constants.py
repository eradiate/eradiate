"""
Mathematical and physical constants.
"""

from .units import unit_registry as ureg

#: Earth radius
EARTH_RADIUS = 6378.1 * ureg.km

#: Lower bound of the default spectral range
SPECTRAL_RANGE_MIN = 250.0 * ureg.nm

#: Upper bound of the default spectral range
SPECTRAL_RANGE_MAX = 3125.0 * ureg.nm
