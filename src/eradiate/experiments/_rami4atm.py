import typing as t
import warnings

import attr

from ._core import EarthObservationExperiment
from ._onedim import measure_inside_atmosphere
from .. import converters, validators
from ..attrs import AUTO, documented, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import OverriddenValueWarning
from ..scenes.atmosphere import Atmosphere, HomogeneousAtmosphere, atmosphere_factory
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.core import KernelDict
from ..scenes.integrators import (
    Integrator,
    PathIntegrator,
    VolPathIntegrator,
    integrator_factory,
)
from ..scenes.measure import Measure, MultiRadiancemeterMeasure, TargetPoint
from ..scenes.measure._core import MeasureFlags
from ..scenes.surface import LambertianSurface, Surface, surface_factory
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


@parse_docs
@attr.s
class Rami4ATMExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and atmosphere.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.

    Notes
    -----
    A post-initialisation step will constrain the measure setup if a
    :class:`.DistantMeasure` is used and no target is defined:

    * if a canopy is defined, the target will be set to the top of the canopy unit cell
      (*i.e.* without its padding);
    * if no canopy is defined, the target will be set according to the atmosphere
      (*i.e.* to [0, 0, `toa`] where `toa` is the top-of-atmosphere altitude);
    * if neither atmosphere nor canopy are defined, the target is set to
      [0, 0, 0].

    This experiment supports arbitrary measure positioning, except for
    :class:`.MultiRadiancemeterMeasure`, for which subsensors are required to
    be either all inside or all outside of the atmosphere. If an unsuitable
    configuration is detected, a :class:`ValueError` will be raised during
    initialisation.

    """

    atmosphere: t.Optional[Atmosphere] = documented(
        attr.ib(
            factory=HomogeneousAtmosphere,
            converter=attr.converters.optional(atmosphere_factory.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Atmosphere)),
        ),
        doc="Atmosphere specification. If set to ``None``, no atmosphere will "
        "be added. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.atmosphere_factory`.",
        type=":class:`.Atmosphere` or None",
        init_type=":class:`.Atmosphere` or dict or None, optional",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    canopy: t.Optional[Canopy] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(biosphere_factory.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Canopy)),
        ),
        doc="Canopy specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.biosphere_factory`.",
        type=":class:`.Canopy` or None",
        init_type=":class:`.Canopy` or dict or None, optional",
        default="None",
    )

    padding: int = documented(
        attr.ib(default=0, converter=int, validator=validators.is_positive),
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

    surface: Surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=surface_factory.convert,
            validator=attr.validators.instance_of(Surface),
        ),
        doc="Surface specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory`.\n"
        "\n"
        ".. note::\n"
        "   Surface size will be overridden using canopy and atmosphere "
        "   parameters, if they are defined.",
        type=":class:`.Surface`",
        init_type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    @surface.validator
    def _surface_validator(self, attribute, value):
        if (self.canopy or self.atmosphere) and value.width is not AUTO:
            warnings.warn(
                OverriddenValueWarning(
                    "user-defined surface width will be overridden by canopy "
                    "or atmosphere size"
                )
            )

    _integrator: Integrator = documented(
        attr.ib(
            default=AUTO,
            converter=converters.auto_or(integrator_factory.convert),
            validator=validators.auto_or(
                attr.validators.instance_of(Integrator),
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

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict()

        # Note: Surface width is always set equal to the largest between the
        # atmosphere and canopy sizes
        if self.atmosphere is not None:
            atm_width = self.atmosphere.kernel_width(ctx)
        else:
            atm_width = 0.0 * ureg.m

        if self.canopy is not None:
            if self.padding > 0:  # We must add extra instances if padding is requested
                canopy = self.canopy.padded_copy(self.padding)
                canopy_width = max(self.canopy.size[:2]) * (2.0 * self.padding + 1.0)
            else:
                canopy = self.canopy
                canopy_width = max(canopy.size[:2])
        else:
            canopy_width = 0.0 * ureg.m
            canopy = None

        scene_width = max(atm_width, canopy_width)
        scene_width = None if scene_width == 0.0 else scene_width
        canopy_width = None if canopy_width == 0.0 else canopy_width
        ctx = ctx.evolve(
            override_scene_width=scene_width, override_canopy_width=canopy_width
        )

        if canopy:
            result.add(canopy, ctx=ctx)

        if self.atmosphere is not None:
            result.add(self.atmosphere, ctx=ctx)
            ctx = ctx.evolve(
                override_scene_width=self.atmosphere.kernel_width(ctx),
            )

            for measure in self.measures:
                if measure_inside_atmosphere(self.atmosphere, measure, ctx):
                    result.add(
                        measure,
                        ctx=ctx.evolve(atmosphere_medium_id=self.atmosphere.id_medium),
                    )
                else:
                    result.add(measure, ctx=ctx)

            result.add(self.surface, self.illumination, self.integrator, ctx=ctx)
        else:
            result.add(
                self.surface,
                self.illumination,
                *self.measures,
                self.integrator,
                ctx=ctx
            )

        return result

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have its ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            # Override ray target location if relevant
            if measure.is_distant():
                if measure.target is None:
                    if self.canopy is not None:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.canopy.size[0],
                            xmax=0.5 * self.canopy.size[0],
                            ymin=-0.5 * self.canopy.size[1],
                            ymax=0.5 * self.canopy.size[1],
                            z=self.canopy.size[2],
                        )
                    else:
                        if self.atmosphere is not None:
                            toa = self.atmosphere.top
                            target_point = [0.0, 0.0, toa.m] * toa.units
                        else:
                            target_point = [0.0, 0.0, 0.0] * ucc.get("length")

                        measure.target = TargetPoint(target_point)

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(Rami4ATMExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result
