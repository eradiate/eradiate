from __future__ import annotations

import logging
import typing as t
import warnings

import attrs

from ._core import EarthObservationExperiment
from ._helpers import (
    check_geometry_atmosphere,
    check_piecewise_compatible,
    measure_inside_atmosphere,
    surface_converter,
)
from ..attrs import AUTO, define, documented
from ..scenes.atmosphere import (
    Atmosphere,
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
    atmosphere_factory,
)
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import SceneElement
from ..scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
)
from ..scenes.integrators import PiecewiseVolPathIntegrator, VolPathIntegrator
from ..scenes.measure import AbstractDistantMeasure, Measure, TargetPoint
from ..scenes.surface import BasicSurface, DEMSurface

logger = logging.getLogger(__name__)


@define
class DEMExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with a digital elevation model (DEM) under a
    1D atmosphere.

    Warnings
    --------
    * Although technically supported, DEMs extending below 0 elevation may be
      a tricky case because atmospheric profile behaviour below sea level is
      undefined. This will be addressed in a future release.

    Notes
    -----
    * When using distant measures, setting a target is highly recommended. This
      experiment will issue a warning during configuration if it detects that a
      distant measure is used with no or an inappropriate target. If a distant
      measure is used and no target is set, it defaults to [0, 0, 0].

    * This experiment supports arbitrary measure positioning, except for
      :class:`.MultiRadiancemeterMeasure`, for which subsensor origins are
      required to be either all inside or all outside of the atmosphere. If an
      unsuitable configuration is detected, a :class:`ValueError` will be raised
      during initialization.

    * Even without an atmosphere, this experiment requries using a volumetric
      path tracing integrator.
    """

    geometry: SceneGeometry = documented(
        attrs.field(
            default="plane_parallel",
            converter=SceneGeometry.convert,
            validator=attrs.validators.instance_of(
                (PlaneParallelGeometry, SphericalShellGeometry)
            ),
        ),
        doc="Problem geometry.",
        type=".SceneGeometry",
        init_type='{"plane_parallel", "spherical_shell"} or dict or '
        ".PlaneParallelGeometry or .SphericalShellGeometry",
        default='"plane_parallel"',
    )

    atmosphere: Atmosphere | None = documented(
        attrs.field(
            factory=HomogeneousAtmosphere,
            converter=attrs.converters.optional(atmosphere_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(Atmosphere)
            ),
        ),
        doc="Atmosphere specification. If set to ``None``, no atmosphere will "
        "be added. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.atmosphere_factory`.",
        type=".Atmosphere or None",
        init_type=".Atmosphere or dict or None",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    surface: BasicSurface | DEMSurface | None = documented(
        attrs.field(
            factory=lambda: BasicSurface(bsdf=LambertianBSDF()),
            converter=attrs.converters.optional(surface_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((BasicSurface, DEMSurface))
            ),
        ),
        doc="Surface specification. If set to ``None``, no surface will be "
        "added. This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory` and :data:`.bsdf_factory`.",
        type=".Surface or None",
        init_type=".BasicSurface or .DEMSurface or .BSDF or dict, optional",
        default=":class:`BasicSurface(bsdf=LambertianBSDF()) <.BasicSurface>`",
    )

    def __attrs_post_init__(self):
        self._normalize_spectral()
        self._normalize_atmosphere()
        self._normalize_measures()
        self._normalize_integrator()

    def _normalize_atmosphere(self) -> None:
        """
        Enforce the experiment geometry on the atmosphere component(s).
        """
        if self.atmosphere is not None:
            # Since 'MolecularAtmosphere' cannot evaluate outside of its
            # vertical extent, we verify here that the experiment's geometry
            # comply with the atmosphere's vertical extent.
            if isinstance(self.atmosphere, MolecularAtmosphere):
                check_geometry_atmosphere(self.geometry, self.atmosphere)
            if isinstance(self.atmosphere, HeterogeneousAtmosphere):
                if self.atmosphere.molecular_atmosphere is not None:
                    check_geometry_atmosphere(
                        self.geometry, self.atmosphere.molecular_atmosphere
                    )

            # Override atmosphere geometry with experiment geometry
            self.atmosphere.geometry = self.geometry

            # The below call to update is required in the case of a
            # HeterogeneousAtmosphere, as it will propagate the geometry
            # override to its components.
            self.atmosphere.update()

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have their ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            # Override ray target location if relevant
            if isinstance(measure, AbstractDistantMeasure):
                if isinstance(self.surface, DEMSurface):
                    if measure.target is None:
                        msg = (
                            f"Measure '{measure.id}' has its target unset "
                            "and the DEM is set. This is not recommended."
                        )

                    elif isinstance(measure.target, TargetPoint):
                        msg = (
                            f"Measure '{measure.id}' uses a point target "
                            "and the DEM is set. This is not recommended."
                        )
                    else:
                        msg = None

                else:
                    if measure.target is None:
                        measure.target = {"type": "point", "xyz": [0, 0, 0]}

                    msg = None

                if msg is not None:
                    warnings.warn(UserWarning(msg))

    def _normalize_integrator(self) -> None:
        """
        Ensures that the integrator is compatible with the atmosphere and geometry.
        """
        piecewise_compatible, msg = check_piecewise_compatible(
            self.geometry, self.atmosphere
        )

        if self.integrator is AUTO:
            if piecewise_compatible:
                self.integrator = PiecewiseVolPathIntegrator()
            else:
                self.integrator = VolPathIntegrator()
        else:
            if (
                isinstance(self.integrator, PiecewiseVolPathIntegrator)
                and not piecewise_compatible
            ):
                raise ValueError(msg)

    def _dataset_metadata(self, measure: Measure) -> dict[str, str]:
        result = super()._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result

    def _context_kwargs(self) -> dict[str, t.Any]:
        kwargs = {}

        for measure in self.measures:
            if measure_inside_atmosphere(self.atmosphere, measure):
                kwargs[f"{measure.sensor_id}.atmosphere_medium_id"] = (
                    self.atmosphere.medium_id
                )

        return kwargs

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        # Inherit docstring

        objects = {}

        # Process atmosphere
        if self.atmosphere is not None:
            objects["atmosphere"] = attrs.evolve(
                self.atmosphere, geometry=self.geometry
            )

        # Process surface
        if self.surface is not None and isinstance(self.surface, BasicSurface):
            objects["surface"] = attrs.evolve(
                self.surface,
                shape=self.geometry.surface_shape,
            )
        else:
            objects["surface"] = self.surface

        objects.update(
            {
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )

        return objects
