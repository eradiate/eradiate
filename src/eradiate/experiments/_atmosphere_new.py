import typing as t

import attrs
import mitsuba as mi

from ._core_new import EarthObservationExperiment, Experiment
from ._helpers import measure_inside_atmosphere, surface_converter
from ..attrs import documented, get_doc, parse_docs
from ..contexts import (
    CKDSpectralContext,
    KernelDictContext,
    MonoSpectralContext,
)
from ..scenes.atmosphere import (
    Atmosphere,
    HomogeneousAtmosphere,
    atmosphere_factory,
)
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import Scene, traverse
from ..scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
)
from ..scenes.integrators import Integrator, VolPathIntegrator, integrator_factory
from ..scenes.measure import DistantMeasure, Measure, TargetPoint
from ..scenes.shapes import RectangleShape, SphereShape
from ..scenes.surface import BasicSurface
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.misc import deduplicate_sorted


@parse_docs
@attrs.define
class AtmosphereExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a one-dimensional scene. This experiment approximates
    a one-dimensional setup using a 3D geometry set up to reproduce the
    effect of invariances typical of 1D geometries. It supports the so-called
    plane parallel and spherical shell geometries.

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
    """

    geometry: t.Union[PlaneParallelGeometry, SphericalShellGeometry] = documented(
        attrs.field(
            default="plane_parallel",
            converter=SceneGeometry.convert,
            validator=attrs.validators.instance_of(
                (PlaneParallelGeometry, SphericalShellGeometry)
            ),
        ),
        doc="Problem geometry.",
        type=".PlaneParallelGeometry or .SphericalShellGeometry",
        init_type="str or dict or .AtmosphereGeometry",
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

    # Override parent
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

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have their ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if isinstance(self.geometry, PlaneParallelGeometry):
                        # Plane parallel geometry: target ground level
                        target_point = [0.0, 0.0, 0.0] * ucc.get("length")

                    elif isinstance(self.geometry, SphericalShellGeometry):
                        # Spherical shell geometry: target ground level
                        target_point = [
                            0.0,
                            0.0,
                            self.geometry.planet_radius.m,
                        ] * self.geometry.planet_radius.units

                    else:  # Shouldn't happen, prevented by validator
                        raise RuntimeError

                    measure.target = TargetPoint(target_point)

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super()._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result

    def contexts(self) -> t.List[KernelDictContext]:
        # Collect contexts from all measures
        sctxs = []
        kwargs = {}

        for measure in self.measures:
            sctxs.extend(measure.spectral_cfg.spectral_ctxs())
            if measure_inside_atmosphere(self.atmosphere, measure):
                kwargs[
                    f"{self.sensor_id}.atmosphere_medium_id"
                ] = self.atmosphere.id_medium

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

        return [KernelDictContext(spectral_ctx=sctx) for sctx in sctxs]

    def init(self) -> None:
        # Create scene
        objects = {}

        # Process atmosphere
        if self.atmosphere is not None:
            objects["atmosphere"] = attrs.evolve(
                self.atmosphere, geometry=self.geometry
            )

        # Process surface
        if self.surface is not None:
            if isinstance(self.geometry, PlaneParallelGeometry):
                width = self.geometry.width
                altitude = (
                    self.atmosphere.bottom
                    if self.atmosphere is not None
                    else 0.0 * ureg.km
                )

                objects["surface"] = attrs.evolve(
                    self.surface,
                    shape=RectangleShape.surface(altitude=altitude, width=width),
                )

            elif isinstance(self.geometry, SphericalShellGeometry):
                altitude = (
                    self.atmosphere.bottom
                    if self.atmosphere is not None
                    else 0.0 * ureg.km
                )

                objects["surface"] = attrs.evolve(
                    self.surface,
                    shape=SphereShape.surface(
                        altitude=altitude, planet_radius=self.geometry.planet_radius
                    ),
                )

            else:  # Shouldn't happen, prevented by validator
                raise RuntimeError

        scene = Scene(
            objects={
                **objects,
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )

        # Generate kernel dictionary and initialise Mitsuba scene
        template, params = traverse(scene)
        kernel_dict = template.render(ctx=KernelDictContext(), drop=True)
        self.mi_scene = mi.load_dict(kernel_dict)
        self.params = params
