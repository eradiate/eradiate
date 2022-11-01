import typing as t

import attrs

from ._atmosphere import measure_inside_atmosphere
from ._core import EarthObservationExperiment
from .. import converters, validators
from ..attrs import AUTO, documented, parse_docs
from ..contexts import KernelDictContext
from ..scenes.atmosphere import (
    Atmosphere,
    AtmosphereGeometry,
    HomogeneousAtmosphere,
    PlaneParallelGeometry,
    SphericalShellGeometry,
    atmosphere_factory,
)
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..scenes.core import KernelDict
from ..scenes.integrators import (
    Integrator,
    PathIntegrator,
    VolPathIntegrator,
    integrator_factory,
)
from ..scenes.measure import Measure, TargetPoint, TargetRectangle
from ..scenes.measure._distant import DistantMeasure
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, CentralPatchSurface, surface_factory
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..util.deprecation import substitute


def _surface_converter(value):
    if isinstance(value, dict):
        try:
            # First, attempt conversion to BSDF
            value = bsdf_factory.convert(value)
        except TypeError:
            # If this doesn't work, attempt conversion to Surface
            return surface_factory.convert(value)

    # If we make it to this point, it means that dict conversion has been
    # performed with success
    if isinstance(value, BSDF):
        return BasicSurface(
            shape=RectangleShape(),
            bsdf=value,
        )

    return value


@parse_docs
@attrs.define
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
    * A post-initialisation step will constrain the measure setup if a
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
      during initialisation.
    """

    geometry: t.Union[PlaneParallelGeometry, SphericalShellGeometry] = documented(
        attrs.field(
            default="plane_parallel",
            converter=AtmosphereGeometry.convert,
            validator=attrs.validators.instance_of(
                (PlaneParallelGeometry, SphericalShellGeometry)
            ),
        ),
        doc="Atmosphere geometry.",
        type=".PlaneParallelGeometry or .SphericalShellGeometry",
        init_type="str or dict or .AtmosphereGeometry",
        default="plane_parallel",
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
        type=".Atmosphere or None",
        init_type=".Atmosphere or dict or None, optional",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    canopy: t.Optional[Canopy] = documented(
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

    surface: t.Union[BasicSurface, CentralPatchSurface, None] = documented(
        attrs.field(
            factory=lambda: BasicSurface(bsdf=LambertianBSDF()),
            converter=attrs.converters.optional(_surface_converter),
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

    _integrator: Integrator = documented(
        attrs.field(
            default=AUTO,
            converter=converters.auto_or(integrator_factory.convert),
            validator=validators.auto_or(
                attrs.validators.instance_of(Integrator),
            ),
        ),
        doc="Monte Carlo integration algorithm specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`."
        "If set to AUTO, the integrator will be set depending on the presence of an atmosphere."
        "If an atmosphere is defined the integrator defaults to :class:`.VolPathIntegrator`"
        "otherwise a :class:`.PathIntegrator` will be used.",
        type=":class:`.Integrator` or AUTO",
        init_type=":class:`.Integrator` or dict or AUTO",
        default="AUTO",
    )

    @property
    def integrator(self) -> Integrator:
        if self._integrator is AUTO:
            if self.atmosphere is not None:
                return VolPathIntegrator()
            else:
                return PathIntegrator()
        return self._integrator

    def __attrs_post_init__(self):
        self._normalize_measures()

    @property
    def _default_surface_width(self):
        return 10.0 * ucc.get("length")

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring

        result = KernelDict()

        # Note: Object size computation logic
        # - The atmosphere, if set, must be the largest object in the
        #   scene. If the geometry setup defines the atmosphere width, it is
        #   used. Otherwise, a size is computed automatically.
        # - The canopy size must be lower that the atmosphere size if it is
        #   defined.
        # - The surface must be larger than the largest object in the scene.
        #   If the atmosphere is set, the surface matches its size.
        #   Otherwise, if the canopy is set, the surface matches its size.
        #   Otherwise, the surface defaults to a size possibly specified by the
        #   geometry setup.

        # Pre-process atmosphere
        if self.atmosphere is not None:
            atmosphere = attrs.evolve(self.atmosphere, geometry=self.geometry)
            atmosphere_width = atmosphere.kernel_width_plane_parallel(ctx)
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
            altitude = atmosphere.bottom if atmosphere is not None else 0.0 * ureg.km
            surface = attrs.evolve(
                self.surface,
                shape=RectangleShape.surface(altitude=altitude, width=surface_width),
            )
        else:
            surface = None

        # Add all configured elements
        if atmosphere is not None:
            result.add(atmosphere, ctx=ctx)

        if canopy is not None:
            result.add(canopy, ctx=ctx)

        if surface is not None:
            result.add(surface, ctx=ctx)

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
                    if (
                        self.canopy is None
                    ):  # No canopy: target single point at ground level
                        measure.target = TargetPoint(
                            [0.0, 0.0, 0.0] * ucc.get("length")
                        )
                    else:  # Canopy: target top of canopy
                        measure.target = TargetRectangle(
                            xmin=-0.5 * self.canopy.size[0],
                            xmax=0.5 * self.canopy.size[0],
                            ymin=-0.5 * self.canopy.size[1],
                            ymax=0.5 * self.canopy.size[1],
                            z=self.canopy.size[2],
                        )

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(CanopyAtmosphereExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result


__getattr__ = substitute(
    {
        "Rami4ATMExperiment": (
            CanopyAtmosphereExperiment,
            {"deprecated_in": "0.22.5", "removed_in": "0.22.7"},
        )
    }
)
