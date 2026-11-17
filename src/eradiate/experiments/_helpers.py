from __future__ import annotations

import logging
import warnings

import attrs

from ..scenes.atmosphere import (
    AbstractHeterogeneousAtmosphere,
    Atmosphere,
    AtmosphereComponent,
    HeterogeneousAtmosphere,
    atmosphere_factory,
)
from ..scenes.atmosphere._core import _DEPRECATED_COMPONENT_PARAMETERS
from ..scenes.bsdfs import BSDF, bsdf_factory
from ..scenes.integrators import Integrator
from ..scenes.measure import (
    AbstractDistantMeasure,
    Measure,
    MultiRadiancemeterMeasure,
    RadiancemeterMeasure,
)
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, Surface, surface_factory

logger = logging.getLogger(__name__)


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

    elif isinstance(measure, AbstractDistantMeasure):
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


def atmosphere_converter(
    value: dict | Atmosphere | AtmosphereComponent | list,
) -> Atmosphere:
    """
    Attempt to convert the atmosphere specification into an atmosphere type.

    Atmospheres can be defined by a single component, in which case it is
    wrapped into a :class:`.HeterogeneousAtmosphere` with the same ID.

    For backward compatibility, medium parameters set on the component are moved
    to the wrapping atmosphere. A list of components is wrapped into a
    :class:`.HeterogeneousAtmosphere` with default settings.
    """
    if isinstance(value, (list, tuple)):
        return HeterogeneousAtmosphere(components=value)

    value = atmosphere_factory.convert(value)

    # Lift parameters from the component and set them to the wrapping
    # HeterogeneousAtmosphere for backward compatibility
    if isinstance(value, AtmosphereComponent):
        params = {
            name: getattr(value, name)
            for name in _DEPRECATED_COMPONENT_PARAMETERS
            if getattr(value, name) is not None
        }
        # Wrap a copy of the component
        component = attrs.evolve(value, **dict.fromkeys(params))
        return HeterogeneousAtmosphere(id=value.id, components=[component], **params)

    return value


def log_piecewise_not_compatible(
    atmosphere: Atmosphere | None, integrator: Integrator
) -> None:
    """
    Log if the atmosphere uses a piecewise medium and the integrator does
    not use the corresponding optimization.
    """

    if (
        isinstance(atmosphere, AbstractHeterogeneousAtmosphere)
        and atmosphere.is_piecewise
        and not integrator.uses_eo_medium_api
    ):
        logger.debug(
            "%s does not use medium tracking: the piecewise medium falls back "
            "to its global majorant",
            type(integrator).__name__,
        )


def warn_ignored_extremum(
    atmosphere: Atmosphere | None, integrator: Integrator
) -> None:
    """
    Warn if the atmosphere requests an extremum structure that will not be
    used, either because its medium is piecewise or because the integrator
    cannot consume it.

    Parameters
    ----------
    atmosphere : Atmosphere
        The scene's atmosphere configuration.

    integrator : Integrator
        The resolved (non-``AUTO``) integrator.
    """
    if not isinstance(atmosphere, HeterogeneousAtmosphere) or all(
        (group.extremum_resolution or atmosphere.extremum_resolution) == (1, 1, 1)
        for group in atmosphere.components_merged
    ):
        return

    if not integrator.uses_eo_medium_api:
        msg = (
            "Extremum structures are not compatible with "
            f"{type(integrator).__name__} and will be ignored; set the "
            "integrator to EOVolpathIntegrator to use them."
        )
    elif atmosphere.is_piecewise:
        msg = (
            "Extremum structures are not used by piecewise media and will be "
            "ignored; set 'force_majorant=True' to use them."
        )
    else:
        return

    warnings.warn(UserWarning(msg), stacklevel=3)


def check_overlap_compatible(
    atmosphere: Atmosphere | None, integrator: Integrator
) -> None:
    """
    Check that the integrator supports overlapping media if the atmosphere is
    rendered as such.

    Parameters
    ----------
    atmosphere : Atmosphere
        The scene's atmosphere configuration.

    integrator : Integrator
        The resolved (non-``AUTO``) integrator.

    Raises
    ------
    ValueError
        If the atmosphere is rendered as overlapping media and the integrator
        does not use the medium API they require.
    """
    if (
        isinstance(atmosphere, HeterogeneousAtmosphere)
        and atmosphere.is_overlapping
        and not integrator.uses_eo_medium_api
    ):
        raise ValueError(
            f"{type(integrator).__name__} does not support overlapping media "
            f"({len(atmosphere.components_merged)} merged components); use "
            "an integrator which uses the medium API (e.g. "
            "EOVolPathIntegrator) or a merge policy producing a single merged "
            "component."
        )


def check_path_compatible(atmosphere: Atmosphere | None) -> tuple[bool, str]:
    """
    Check if the scene is compatible with the ``path`` integrator.

    Parameters
    ----------
    atmosphere : Atmosphere
        The scene's atmosphere configuration.

    Returns
    -------
    tuple[bool, str]
        Path compatible flag and an exception message if the flag is not raised.
        The message is empty if the flag is raised.
    """

    if atmosphere is not None:
        debug_message = (
            "You have an atmosphere in your scene, prefer using the Path Integrator."
        )
        return False, debug_message

    return True, ""
