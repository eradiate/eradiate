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
from ..scenes.measure import Measure, TargetPoint
from ..scenes.surface import LambertianSurface, Surface, surface_factory
from ..units import unit_context_config as ucc


@parse_docs
@attr.s
class OneDimExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a one-dimensional scene. This experiment approximates
    a one-dimensional setup using a 3D geometry set up so as to reproduce the
    effect of translational invariance.

    Notes
    -----
    A post-initialisation step will constrain the measure setup if a
    :class:`.DistantMeasure` is used and no target is defined:

    * if an atmosphere is defined, the target will be set to [0, 0, TOA];
    * if no atmosphere is defined, the target will be set to [0, 0, 0].
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
                OverriddenValueWarning("surface size will be overridden by atmosphere")
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

        if self.atmosphere is not None:
            result.add(self.atmosphere, ctx=ctx)
            ctx = ctx.evolve(override_scene_width=self.atmosphere.kernel_width(ctx))

        result.add(
            self.surface, self.illumination, *self.measures, self.integrator, ctx=ctx
        )

        return result

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure target and origin are set to appropriate
        values. Processed measures will have its ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if self.atmosphere is not None:
                        toa = self.atmosphere.top
                        target_point = [0.0, 0.0, toa.m] * toa.units
                    else:
                        target_point = [0.0, 0.0, 0.0] * ucc.get("length")

                    measure.target = TargetPoint(target_point)

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(OneDimExperiment, self)._dataset_metadata(measure)

        if isinstance(measure, DistantMeasure):
            result["title"] = "Top-of-atmosphere simulation results"

        return result
