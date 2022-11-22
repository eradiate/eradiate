from __future__ import annotations

import typing as t
import warnings

import attrs

from ._core import EarthObservationExperiment, Experiment
from ._helpers import measure_inside_atmosphere, _surface_converter
from ..attrs import AUTO, documented, get_doc, parse_docs
from ..contexts import KernelDictContext
from ..scenes.atmosphere import (
    Atmosphere,
    HomogeneousAtmosphere,
    PlaneParallelGeometry,
    atmosphere_factory,
)
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import KernelDict
from ..scenes.integrators import Integrator, VolPathIntegrator, integrator_factory
from ..scenes.measure import Measure, TargetPoint
from ..scenes.measure._distant import DistantMeasure
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, DEMSurface, surface_factory
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


@parse_docs
@attrs.define
class DEMExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with a digital elevation model (DEM). 
    This experiment approximates a one-dimensional setup using 
    a 3D geometry set up to reproduce the effect of invariances typical 
    of 1D geometries.

    Notes
    -----
    * A post-initialisation step will constrain the measure setup if a
      distant measure is used and no target is defined:

      * if an atmosphere is defined, the target will be set to [0, 0, TOA];
      * if no atmosphere is defined, the target will be set to [0, 0, 0].

    * This experiment supports arbitrary measure positioning, except for
      :class:`.MultiRadiancemeterMeasure`, for which subsensor origins are
      required to be either all inside or all outside of the atmosphere. If an
      unsuitable configuration is detected, a :class:`ValueError` will be raised
      during initialisation.

    * Currently this experiment is limited to plane-parallel atmospheric geometry.
    """

    atmosphere: t.Optional[Atmosphere] = documented(
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
        type=":class:`.Atmosphere` or None",
        init_type=":class:`.Atmosphere` or dict or None",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    surface: t.Optional[BasicSurface] = documented(
        attrs.field(
            factory=lambda: BasicSurface(bsdf=LambertianBSDF()),
            converter=attrs.converters.optional(_surface_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(BasicSurface)
            ),
        ),
        doc="Surface specification. If set to ``None``, no surface will be "
        "added. This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory` and :data:`.bsdf_factory`.",
        type=".BasicSurface or None",
        init_type=".BasicSurface or .BSDF or dict, optional",
        default=":class:`BasicSurface(bsdf=LambertianBSDF()) <.BasicSurface>`",
    )

    dem: t.Optional[DEMSurface] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(surface_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(DEMSurface)
            ),
        ),
        doc="Digital elevation model (DEM) specification. If set to ``None``, no DEM will be "
        "added. This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory`",
        type=".DEMSurface or None",
        init_type=".DEMSurface or dict, optional",
        default="None",
    )

    _integrator: Integrator = documented(
        attrs.field(
            factory=VolPathIntegrator,
            converter=integrator_factory.convert,
            validator=attrs.validators.instance_of(Integrator),
        ),
        doc=get_doc(Experiment, attrib="_integrator", field="doc"),
        type=get_doc(Experiment, attrib="_integrator", field="type"),
        init_type=get_doc(Experiment, attrib="_integrator", field="init_type"),
        default=":class:`VolPathIntegrator() <.VolPathIntegrator>`",
    )

    def __attrs_post_init__(self):
        self._normalize_measures()

    @property
    def _default_surface_width(self):
        return 1.0 * ureg.km

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict({"type": "scene"})

        # Process atmosphere
        if self.atmosphere is not None:
            atmosphere = attrs.evolve(self.atmosphere, geometry=PlaneParallelGeometry())
            result.add(atmosphere, ctx=ctx)
        else:
            atmosphere = None

        # Process surface
        if self.surface is not None:
            if atmosphere is not None:
                width = atmosphere.kernel_width_plane_parallel(ctx)
                altitude = atmosphere.bottom
            else:
                width = self._default_surface_width
                altitude = 0.0 * ureg.km

            surface = attrs.evolve(
                self.surface,
                shape=RectangleShape.surface(altitude=altitude, width=width),
            )

            result.add(surface, ctx=ctx)

        # Process DEM
        if self.dem is not None:
            for measure in self.measures:
                if isinstance(measure.target, TargetPoint):
                    warnings.warn(
                        UserWarning(
                            f"Your measure {measure.id}, uses a point target. "
                            f"This might be undesirable when simulating a DEM."
                        )
                    )

            result.add(self.dem, ctx=ctx)

        # Process measures
        for measure in self.measures:
            if measure_inside_atmosphere(atmosphere, measure, ctx):
                result.add(
                    measure,
                    ctx=ctx.evolve(atmosphere_medium_id=self.atmosphere.id_medium),
                )
            else:
                result.add(measure, ctx=ctx)

        # Process illumination
        result.add(self.illumination, ctx=ctx)

        # Process integrator
        result.add(self.integrator, ctx=ctx)

        return result

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have its ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    target_point = [0.0, 0.0, 0.0] * ucc.get("length")
                    measure.target = TargetPoint(target_point)

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(DEMExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result
