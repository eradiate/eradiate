import typing as t
import warnings

import attr

from ._core import EarthObservationExperiment, Experiment
from ..attrs import AUTO, documented, get_doc, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import OverriddenValueWarning
from ..scenes.atmosphere import Atmosphere, HomogeneousAtmosphere, atmosphere_factory
from ..scenes.core import KernelDict
from ..scenes.integrators import Integrator, VolPathIntegrator, integrator_factory
from ..scenes.measure import Measure, MultiRadiancemeterMeasure, TargetPoint
from ..scenes.measure._core import MeasureFlags
from ..scenes.surface import LambertianSurface, Surface, surface_factory
from ..units import unit_context_config as ucc


def measure_inside_atmosphere(atmosphere, measure, ctx):
    """
    Evaluate whether a sensor is placed within an atmosphere.

    Raises a ValueError if called with a :class:`MultiRadiancemeterMeasure` with
    origins both inside and outside of the atmosphere.
    """
    bbox = atmosphere.eval_bbox(ctx)

    if isinstance(measure, MultiRadiancemeterMeasure):
        inside = [bbox.contains(origin) for origin in measure.origins]
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
    elif measure.flags & MeasureFlags.DISTANT:
        return False
    else:
        return bbox.contains(measure.origin)


@parse_docs
@attr.s
class OneDimExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a one-dimensional scene. This experiment approximates
    a one-dimensional setup using a 3D geometry set up so as to reproduce the
    effect of translational invariance.

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
        init_type=":class:`.Atmosphere` or dict or None",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    surface: Surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=surface_factory.convert,
            validator=attr.validators.instance_of(Surface),
        ),
        doc="Surface specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory`.",
        type=":class:`.Surface`",
        init_type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    @surface.validator
    def _surface_validator(self, attribute, value):
        if self.atmosphere and value.width is not AUTO:
            warnings.warn(
                OverriddenValueWarning(
                    "user-defined surface width will be overridden by "
                    "atmosphere width"
                )
            )

    _integrator: Integrator = documented(
        attr.ib(
            factory=VolPathIntegrator,
            converter=integrator_factory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc=get_doc(Experiment, attrib="_integrator", field="doc"),
        type=get_doc(Experiment, attrib="_integrator", field="type"),
        init_type=get_doc(Experiment, attrib="_integrator", field="init_type"),
        default=":class:`VolPathIntegrator() <.VolPathIntegrator>`",
    )

    def __attrs_post_init__(self):
        self._normalize_measures()

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict({"type": "scene"})

        # Note: Surface width is always set equal to atmosphere width
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
        Ensure that distant measure target and origin are set to appropriate
        values. Processed measures will have its ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            if measure.is_distant():
                if measure.target is None:
                    if self.atmosphere is not None:
                        toa = self.atmosphere.top
                        target_point = [0.0, 0.0, toa.m] * toa.units
                    else:
                        target_point = [0.0, 0.0, 0.0] * ucc.get("length")

                    measure.target = TargetPoint(target_point)

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(OneDimExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result
