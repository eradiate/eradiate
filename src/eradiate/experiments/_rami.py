import typing as t
import warnings

import attr

from ._core import EarthObservationExperiment, Experiment
from .. import supported_mode, validators
from .._mode import ModeFlags
from ..attrs import AUTO, documented, get_doc, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import OverriddenValueWarning
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.core import KernelDict
from ..scenes.integrators import Integrator, PathIntegrator, integrator_factory
from ..scenes.measure import Measure
from ..scenes.measure._core import MeasureFlags
from ..scenes.surface import LambertianSurface, Surface, surface_factory


@parse_docs
@attr.s
class RamiExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and no atmosphere.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.

    Notes
    -----
    A post-initialisation step will constrain the measure setup if a
    :class:`.DistantMeasure` is used and no target is defined:

    * if a canopy is defined, the target will be set to the top of the canopy unit cell
      (*i.e.* without its padding);
    * if no canopy is defined, the target will be set to [0, 0, 0].
    """

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
        init_type=":class:`.Canopy` or dict or None",
        default="None",
    )

    padding: int = documented(
        attr.ib(default=0, converter=int, validator=validators.is_positive),
        doc="Padding level. The scene will be padded with copies to account for "
        "adjacency effects. This, in practice, has effects similar to "
        "making the scene periodic."
        "A value of 0 will yield only the defined scene. A value of 1 "
        "will add one copy in every direction, yielding a 3×3 patch. A "
        "value of 2 will yield a 5×5 patch, etc. The optimal padding level "
        "depends on the scene.",
        type="int",
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
        "   Surface size will be overridden using canopy parameters.",
        type=":class:`.Surface`",
        init_type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    @surface.validator
    def _surface_validator(self, attribute, value):
        if self.canopy and value.width is not AUTO:
            warnings.warn(
                OverriddenValueWarning("surface size will be overridden by canopy")
            )

    _integrator: Integrator = documented(
        attr.ib(
            factory=PathIntegrator,
            converter=integrator_factory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc=get_doc(Experiment, attrib="_integrator", field="doc"),
        type=get_doc(Experiment, attrib="_integrator", field="type"),
        init_type=get_doc(Experiment, attrib="_integrator", field="init_type"),
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    def __attrs_pre_init__(self):
        # Only tested with monochromatic modes
        supported_mode(ModeFlags.ANY_MONO)

    def __attrs_post_init__(self):
        self._normalize_measures()

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict()

        if self.canopy is not None:
            if self.padding > 0:  # We must add extra instances if padding is requested
                canopy = self.canopy.padded_copy(self.padding)
                ctx = ctx.evolve(
                    override_scene_width=max(self.canopy.size[:2])
                    * (2.0 * self.padding + 1.0)
                )
            else:
                canopy = self.canopy
                ctx = ctx.evolve(override_scene_width=max(self.canopy.size[:2]))

            result.add(canopy, ctx=ctx)

        result.add(
            self.surface,
            self.illumination,
            *self.measures,
            self.integrator,
            ctx=ctx,
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
                        ctx = KernelDictContext()
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.surface.kernel_width(ctx),
                            xmax=0.5 * self.surface.kernel_width(ctx),
                            ymin=-0.5 * self.surface.kernel_width(ctx),
                            ymax=0.5 * self.surface.kernel_width(ctx),
                        )

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(RamiExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-canopy simulation results"

        return result
