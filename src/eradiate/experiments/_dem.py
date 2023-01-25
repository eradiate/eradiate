import typing as t
import warnings

import attrs

from ._core import EarthObservationExperiment, Experiment
from ._helpers import measure_inside_atmosphere, surface_converter
from ..attrs import documented, get_doc, parse_docs
from ..contexts import (
    CKDSpectralContext,
    KernelDictContext,
    MonoSpectralContext,
    SpectralContext,
)
from ..scenes.atmosphere import Atmosphere, HomogeneousAtmosphere, atmosphere_factory
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import Scene
from ..scenes.geometry import PlaneParallelGeometry, SceneGeometry
from ..scenes.integrators import Integrator, VolPathIntegrator, integrator_factory
from ..scenes.measure import DistantMeasure, Measure, TargetPoint
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, DEMSurface, surface_factory
from ..units import unit_registry as ureg
from ..util.misc import deduplicate_sorted


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
            converter=attrs.converters.optional(surface_converter),
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
        self._normalize_atmosphere()
        self._normalize_measures()

    def _normalize_atmosphere(self) -> None:
        """
        Ensure consistency between the atmosphere and experiment geometries.
        """
        if self.atmosphere is not None:
            self.atmosphere.geometry = self.geometry

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have their ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            # Override ray target location if relevant
            if isinstance(measure, DistantMeasure):
                if self.dem is not None:
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
                        msg = (
                            f"Measure '{measure.id}' has its target unset. "
                            "This is not recommended. Forcing to [0, 0, 0]."
                        )
                        measure.target = {"type": "point", "xyz": [0, 0, 0]}

                    else:
                        msg = None

                if msg is not None:
                    warnings.warn(UserWarning(msg))

    @property
    def _default_surface_width(self):
        return 1.0 * ureg.km

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super()._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result

    @property
    def _context_kwargs(self) -> t.Dict[str, t.Any]:
        kwargs = {}

        for measure in self.measures:
            if measure_inside_atmosphere(self.atmosphere, measure):
                kwargs[
                    f"{measure.sensor_id}.atmosphere_medium_id"
                ] = self.atmosphere.medium_id

        return kwargs

    @property
    def contexts(self) -> t.List[KernelDictContext]:
        # Inherit docstring

        # Collect contexts from all measures
        sctxs = []

        for measure in self.measures:
            sctxs.extend(measure.spectral_cfg.spectral_ctxs())

        # Sort and remove duplicates
        key = {
            MonoSpectralContext: lambda sctx: sctx.wavelength.m,
            CKDSpectralContext: lambda sctx: (
                sctx.bindex.bin.wcenter.m,
                sctx.bindex.index,
            ),
        }[type(sctxs[0])]

        sctxs = deduplicate_sorted(
            sorted(sctxs, key=key), cmp=lambda x, y: key(x) == key(y)
        )
        kwargs = self._context_kwargs

        return [KernelDictContext(spectral_ctx=sctx, kwargs=kwargs) for sctx in sctxs]

    @property
    def context_init(self) -> KernelDictContext:
        # Inherit docstring

        return KernelDictContext(
            spectral_ctx=SpectralContext.new(), kwargs=self._context_kwargs
        )

    @property
    def scene(self) -> Scene:
        # Inherit docstring

        objects = {}

        # Process atmosphere
        if self.atmosphere is not None:
            objects["atmosphere"] = self.atmosphere

        # Process surface
        if self.surface is not None:
            if self.atmosphere is not None:
                surface_width = self.atmosphere.geometry.width
                surface_altitude = self.atmosphere.bottom
            else:
                surface_width = self._default_surface_width
                surface_altitude = 0.0 * ureg.km

            objects["surface"] = attrs.evolve(
                self.surface,
                shape=RectangleShape.surface(
                    altitude=surface_altitude,
                    width=surface_width,
                ),
            )

        # Process DEM
        if self.dem is not None:
            result.add(self.dem, ctx=ctx)

        return Scene(
            objects={
                **objects,
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )
