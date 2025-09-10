from __future__ import annotations

import logging
import typing as t

import attrs

from ._core import EarthObservationExperiment
from ._helpers import (
    check_geometry_atmosphere,
    check_path_compatible,
    check_piecewise_compatible,
    measure_inside_atmosphere,
    surface_converter,
)
from .. import validators
from ..attrs import AUTO, define, documented
from ..scenes.atmosphere import (
    Atmosphere,
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
    atmosphere_factory,
)
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import SceneElement
from ..scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
)
from ..scenes.integrators import (
    PathIntegrator,
    PiecewiseVolPathIntegrator,
    VolPathIntegrator,
)
from ..scenes.measure import AbstractDistantMeasure, Measure
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, CentralPatchSurface
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@define
class CanopyAtmosphereExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and atmosphere.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.

    Warnings
    --------
    * Canopy padding is controlled using the `padding` parameter: do *not* pad
      the canopy itself manually.

    Notes
    -----
    * A post-initialization step will constrain the measure setup if a
      distant measure is used and no target is defined:

      * if a canopy is defined, the target will be set to the top of the canopy
        unit cell (*i.e.* without its padding);
      * if no canopy is defined, the target will be set according to the
        atmosphere (*i.e.* to [0, 0, `toa`] where `toa` is the top-of-atmosphere
        altitude);
      * if neither atmosphere nor canopy are defined, the target is set to
        [0, 0, 0].

    * This experiment supports arbitrary measure positioning, except for
      :class:`.MultiRadiancemeterMeasure`, for which subsensor origins are
      required to be either all inside or all outside of the atmosphere. If an
      unsuitable configuration is detected, a :class:`ValueError` will be raised
      during initialization.

    * Currently this experiment is limited to the plane-parallel geometry.
    """

    # Currently, only the plane parallel geometry is supported
    geometry: PlaneParallelGeometry = documented(
        attrs.field(
            default="plane_parallel",
            converter=SceneGeometry.convert,
            validator=attrs.validators.instance_of(PlaneParallelGeometry),
        ),
        doc="Problem geometry.",
        type=".PlaneParallelGeometry",
        init_type="str or dict or .PlaneParallelGeometry",
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
        init_type=".Atmosphere or dict or None, optional",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    canopy: Canopy | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(biosphere_factory.convert),
            validator=attrs.validators.optional(attrs.validators.instance_of(Canopy)),
        ),
        doc="Canopy specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.biosphere_factory`.",
        type=".Canopy or None",
        init_type=".Canopy or dict or None, optional",
        default="None",
    )

    padding: int = documented(
        attrs.field(default=0, converter=int, validator=validators.is_positive),
        doc="Padding level. The canopy will be padded with copies to account for "
        "adjacency effects. This, in practice, has effects similar to "
        "making the scene periodic."
        "A value of 0 will yield only the defined scene. A value of 1 "
        "will add one copy in every direction, yielding a 3×3 patch. A "
        "value of 2 will yield a 5×5 patch, etc. The optimal padding level "
        "depends on the scene.",
        type="int",
        init_type="int, optional",
        default="0",
    )

    surface: BasicSurface | CentralPatchSurface | None = documented(
        attrs.field(
            factory=lambda: BasicSurface(bsdf=LambertianBSDF()),
            converter=attrs.converters.optional(surface_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((BasicSurface, CentralPatchSurface))
            ),
        ),
        doc="Surface specification. A :class:`.Surface` object may be passed: "
        "its shape specifications will be bypassed and the surface size will "
        "be computed automatically upon kernel dictionary generation. "
        "A :class:`.BSDF` may also be passed: it will be wrapped automatically "
        "in a :class:`.BasicSurface` instance. If a dictionary is passed, it "
        "will be first interpreted as a :class:`.BSDF`; if this fails, it will "
        "then be interpreted as a :class:`.Surface`. Finally, this field can "
        "be set to ``None``: in that case, no surface will be added.",
        type=".Surface or None",
        init_type=".Surface or .BSDF or dict or None, optional",
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
                if measure.target is None:
                    if self.canopy is None:  # No canopy: target origin point
                        measure.target = {"type": "point", "xyz": [0, 0, 0]}

                    else:  # Canopy: target top of canopy
                        measure.target = {
                            "type": "rectangle",
                            "xmin": -0.5 * self.canopy.size[0],
                            "xmax": 0.5 * self.canopy.size[0],
                            "ymin": -0.5 * self.canopy.size[1],
                            "ymax": 0.5 * self.canopy.size[1],
                            "z": self.canopy.size[2],
                        }

    def _normalize_integrator(self) -> None:
        """
        Ensures that the integrator is compatible with the atmosphere and geometry.
        """
        piecewise_compatible, pw_msg = check_piecewise_compatible(
            self.geometry, self.atmosphere
        )
        path_compatible, path_msg = check_path_compatible(self.atmosphere)

        if self.integrator is AUTO:
            if piecewise_compatible:
                self.integrator = PiecewiseVolPathIntegrator()
            else:
                self.integrator = VolPathIntegrator()
        else:
            msg = ""
            if (
                isinstance(self.integrator, PiecewiseVolPathIntegrator)
                and not piecewise_compatible
            ):
                msg = pw_msg

            if isinstance(self.integrator, PathIntegrator) and not path_compatible:
                msg = path_msg

            if msg:
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
    def _default_surface_width(self):
        return 10.0 * ucc.get("length")

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        # Inherit docstring

        objects = {}

        # Note: Object size computation logic
        # - The atmosphere, if set, must be the largest object in the
        #   scene. If the geometry setup defines the atmosphere width, it is
        #   used. Otherwise, a size is computed automatically.
        # - The canopy size must be lower than the atmosphere size if it is
        #   defined.
        # - The surface must be larger than the largest object in the scene.
        #   If the atmosphere is set, the surface matches its size.
        #   Otherwise, if the canopy is set, the surface matches its size.
        #   Otherwise, the surface defaults to a size possibly specified by the
        #   geometry setup.

        # Pre-process atmosphere
        if self.atmosphere is not None:
            atmosphere = attrs.evolve(self.atmosphere, geometry=self.geometry)
            atmosphere_width = self.geometry.width
        else:
            atmosphere = None
            atmosphere_width = 0.0 * ureg.m

        # Pre-process canopy
        if self.canopy is not None:
            canopy_width = max(self.canopy.size[:2])

            if self.padding > 0:  # We must add extra instances if padding is requested
                canopy_width *= 2.0 * self.padding + 1.0
                canopy = self.canopy.padded_copy(self.padding)
            else:
                canopy = self.canopy
        else:
            canopy = None
            canopy_width = 0.0 * ureg.m

        # Check sizes, compute surface size
        if atmosphere is not None:
            assert atmosphere_width > canopy_width
        surface_width = self._default_surface_width
        if canopy_width > surface_width:
            surface_width = canopy_width
        if atmosphere_width > surface_width:
            surface_width = atmosphere_width

        # Pre-process surface
        if self.surface is not None:
            altitude = (
                atmosphere.bottom_altitude if atmosphere is not None else 0.0 * ureg.km
            )
            surface = attrs.evolve(
                self.surface,
                shape=RectangleShape.surface(altitude=altitude, width=surface_width),
            )
        else:
            surface = None

        # Add all configured elements to the scene
        if atmosphere is not None:
            objects["atmosphere"] = atmosphere

        if canopy is not None:
            objects["canopy"] = canopy

        if surface is not None:
            objects["surface"] = surface

        objects.update(
            {
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )

        return objects
