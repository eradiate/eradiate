import warnings
from typing import Optional

import attr

from ..core._scene import Scene
from ..._mode import ModeFlags, supported_mode
from ...attrs import AUTO, documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...exceptions import OverriddenValueWarning
from ...scenes.atmosphere import Atmosphere, HomogeneousAtmosphere, atmosphere_factory
from ...scenes.core import KernelDict
from ...scenes.integrators import Integrator, VolPathIntegrator, integrator_factory
from ...scenes.measure._distant import (
    DistantMeasure,
    TargetOriginPoint,
    TargetOriginSphere,
)
from ...scenes.surface import LambertianSurface, Surface, surface_factory
from ...units import unit_context_config as ucc


@parse_docs
@attr.s
class OneDimScene(Scene):
    """
    Scene abstraction suitable for radiative transfer simulation on
    one-dimensional scenes.
    """

    atmosphere: Optional[Atmosphere] = documented(
        attr.ib(
            factory=HomogeneousAtmosphere,
            converter=attr.converters.optional(atmosphere_factory.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Atmosphere)),
        ),
        doc="Atmosphere specification. If set to ``None``, no atmosphere will "
        "be added. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.atmosphere_factory`.",
        type=":class:`.Atmosphere` or dict or None",
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
        "interpreted by "
        ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`.",
        type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    @surface.validator
    def _surface_validator(self, attribute, value):
        if self.atmosphere and value.width is not AUTO:
            warnings.warn(
                OverriddenValueWarning("surface size will be overridden by atmosphere")
            )

    integrator: Integrator = documented(
        attr.ib(
            factory=VolPathIntegrator,
            converter=integrator_factory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc=get_doc(Scene, attrib="integrator", field="doc"),
        type=get_doc(Scene, attrib="integrator", field="type"),
        default=":class:`VolPathIntegrator() <.VolPathIntegrator>`",
    )

    def __attrs_pre_init__(self):
        # Only tested with monochromatic and CKD modes
        supported_mode(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD)

    def update(self):
        # Parts of the init sequence we could take care of using converters
        # TODO: This is not robust and will surely break if any update of
        #  modified attributes is attempted after object creation.
        #  Solution: Add these steps to a preprocess step, leave it out of the
        #  post-init sequence.

        # Don't forget to call super class post-init
        super(OneDimScene, self).update()

        # Process measures
        for measure in self.measures:
            # Override ray target and origin if relevant
            # Likely deserves more polishing
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if self.atmosphere is not None:
                        toa = self.atmosphere.top
                        target_point = [0.0, 0.0, toa.m] * toa.units
                    else:
                        target_point = [0.0, 0.0, 0.0] * ucc.get("length")

                    measure.target = TargetOriginPoint(target_point)

                if measure.origin is None:
                    radius = (
                        self.atmosphere.top / 100.0
                        if self.atmosphere is not None
                        else 1.0 * ucc.get("length")
                    )

                    measure.origin = TargetOriginSphere(
                        center=measure.target.xyz, radius=radius
                    )

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict.new(ctx=ctx)

        if self.atmosphere is not None:
            result.add(self.atmosphere, ctx=ctx)
            ctx = ctx.evolve(override_scene_width=self.atmosphere.kernel_width(ctx))

        result.add(
            self.surface, self.illumination, *self.measures, self.integrator, ctx=ctx
        )

        return result
