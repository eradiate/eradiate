from __future__ import annotations

from ..scenes.atmosphere import Atmosphere
from ..scenes.bsdfs import BSDF, bsdf_factory
from ..scenes.measure import (
    DistantMeasure,
    Measure,
    MultiRadiancemeterMeasure,
    RadiancemeterMeasure,
)
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, Surface, surface_factory


def measure_inside_atmosphere(atmosphere: Atmosphere, measure: Measure) -> bool:
    """
    Evaluate whether a sensor is placed within an atmosphere.

    Raises a ValueError if called with a :class:`.MultiRadiancemeterMeasure`
    with origins both inside and outside the atmosphere.
    """
    if atmosphere is None:
        return False

    shape = atmosphere.shape

    if isinstance(measure, MultiRadiancemeterMeasure):
        inside = shape.contains(measure.origins)

        if all(inside):
            return True

        elif not any(inside):
            return False

        else:
            raise ValueError(
                "Inconsistent placement of MultiRadiancemeterMeasure origins. "
                "Origins must lie either all inside or all outside of the "
                "atmosphere."
            )

    elif isinstance(measure, DistantMeasure):
        # Note: This will break if the user makes something weird such as using
        # a large offset value which would put some origins outside and others
        # inside the atmosphere shape
        return not measure.is_distant()

    elif isinstance(measure, RadiancemeterMeasure):
        return shape.contains(measure.origin)

    else:
        # Note: This will likely break if a new measure type is added
        return shape.contains(measure.origin)


def surface_converter(value: dict | Surface | BSDF) -> Surface:
    """
    Attempt to convert the surface specification into a surface type.

    Surfaces can be defined purely by their BSDF, in which case Eradiate will
    define a rectangular surface and attach that BSDF to it.
    """
    if isinstance(value, dict):
        try:
            # First, attempt conversion to BSDF
            value = bsdf_factory.convert(value)
        except ValueError:  # Type ID could not be found in BSDF factory registry
            # Attempt conversion to Surface
            return surface_factory.convert(value)

    # If we make it to this point, it means that dict conversion has been
    # performed with success
    if isinstance(value, BSDF):
        return BasicSurface(
            shape=RectangleShape(),
            bsdf=value,
        )

    return value
