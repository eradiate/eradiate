"""
Mathematical and physical constants.
"""

from .units import unit_registry as ureg

#: Earth radius
EARTH_RADIUS = 6378.1 * ureg.km
