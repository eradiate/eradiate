from __future__ import annotations

from ..scenes.atmosphere import (
    AbstractHeterogeneousAtmosphere,
    Atmosphere,
    MolecularAtmosphere,
)
from ..scenes.bsdfs import BSDF, bsdf_factory
from ..scenes.geometry import SceneGeometry, SphericalShellGeometry
from ..scenes.measure import (
    AbstractDistantMeasure,
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


def check_geometry_atmosphere(
    geometry: SceneGeometry, atmosphere: MolecularAtmosphere
) -> None:
    """
    Check that the experiment geometry is compatible with the vertical extent of
    the molecular atmosphere.

    Parameters
    ----------
    geometry : SceneGeometry
        An experiment geometry.

    atmosphere : MolecularAtmosphere
        A molecular atmosphere.

    Raises
    ------
    ValueError
        If the geometry vertical extent exceeds the atmosphere vertical
        extent.
    """

    radprops_zbounds = atmosphere.radprops_profile.zbounds
    geometry_zbounds = geometry.zgrid.levels[[0, -1]]
    suggested_solution = (
        "Try to set the experiment geometry so that it does not go beyond "
        "the vertical extent of the molecular atmosphere."
    )
    if (geometry_zbounds[0] < radprops_zbounds[0]) or (
        geometry_zbounds[1] > radprops_zbounds[1]
    ):
        raise ValueError(
            "Attribtues 'geometry' and 'atmosphere' are incompatible: "
            f"'geometry.zgrid' bounds ({geometry_zbounds}) go beyond the "
            f"bounds of 'atmosphere.radprops_profile.zbounds' ({radprops_zbounds}). "
            f"{suggested_solution}"
        )


def check_piecewise_compatible(
    geometry: SceneGeometry, atmosphere: Atmosphere | None
) -> tuple[bool, str]:
    """
    Check that the piecewise integrator is compatible with the experiment's
    atmosphere and geometry.

    Parameters
    ----------
    geometry : SceneGeometry
        An experiment geometry.

    atmosphere : Atmosphere
        The scene's atmosphere configuration.

    Returns
    -------
    tuple[bool, str]
        Piecewise compatible flag and an exception message if the flag is not raised.
        The message is empty if the flag is raised.
    """

    if isinstance(geometry, SphericalShellGeometry):
        debug_message = (
            "Piecewise integrator not compatible with spherical shell geometry."
        )
        return False, debug_message

    if atmosphere is None or not isinstance(
        atmosphere, AbstractHeterogeneousAtmosphere
    ):
        debug_message = f"Piecewise integrator not compatible with medium type : {type(atmosphere)}."
        return False, debug_message

    if atmosphere.force_majorant:
        debug_message = "Piecewise integrator not compatible with a medium that has force_majorant = True."
        return False, debug_message

    return True, ""


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
