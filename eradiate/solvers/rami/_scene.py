from typing import Optional

import attr

from ..core._scene import Scene
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...scenes.biosphere import BiosphereFactory, Canopy
from ...scenes.core import KernelDict
from ...scenes.integrators import Integrator, IntegratorFactory, PathIntegrator
from ...scenes.measure import DistantMeasure
from ...scenes.surface import LambertianSurface, Surface, SurfaceFactory


@parse_docs
@attr.s
class RamiScene(Scene):
    """
    Scene abstraction suitable for radiative transfer simulation on RAMI
    benchmark scenes.
    """

    surface: Surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=SurfaceFactory.convert,
            validator=attr.validators.instance_of(Surface),
        ),
        doc="Surface specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`. "
        ".. note:: Surface size will be overridden using canopy parameters.",
        type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`",
    )

    canopy: Canopy = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(BiosphereFactory.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Canopy)),
        ),
        doc="Canopy specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`BiosphereFactory.convert() <.BiosphereFactory.convert>`.",
        type=":class:`.Canopy` or dict",
        default=":class:`HomogeneousDiscreteCanopy() <.HomogeneousDiscreteCanopy>`",
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

    integrator: Integrator = documented(
        attr.ib(
            factory=PathIntegrator,
            converter=IntegratorFactory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc=get_doc(Scene, attrib="integrator", field="doc"),
        type=get_doc(Scene, attrib="integrator", field="type"),
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    def update(self):
        # Parts of the init sequence we couldn't take care of using converters
        # TODO: This is not robust and will surely break if any update of
        #  modified attributes is attempted after object creation.
        #  Solution: Add these steps to a preprocess step, leave it out of the
        #  post-init sequence.

        # Don't forget to call super class post-init
        super(RamiScene, self).update()

        # Process measures
        for measure in self.measures:
            # Override ray target location if relevant
            if isinstance(measure, DistantMeasure):
                if measure.target is None:
                    if self.canopy is not None:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.canopy.size[0],
                            xmax=0.5 * self.canopy.size[0],
                            ymin=-0.5 * self.canopy.size[1],
                            ymax=0.5 * self.canopy.size[1],
                        )
                    else:
                        measure.target = dict(
                            type="rectangle",
                            xmin=-0.5 * self.surface.kernel_width(),
                            xmax=0.5 * self.surface.kernel_width(),
                            ymin=-0.5 * self.surface.kernel_width(),
                            ymax=0.5 * self.surface.kernel_width(),
                        )

    def kernel_dict(self, ctx: KernelDictContext = None):
        result = KernelDict.new()

        if self.canopy is not None:
            if self.padding > 0:  # We must add extra instances if padding is requested
                canopy = self.canopy.padded(self.padding)
                ctx.override_surface_width = (
                    max(self.canopy.size[:2]) * 2.0 * self.padding + 1.0
                )
            else:
                canopy = self.canopy
                ctx.override_surface_width = max(self.canopy.size[:2])

            result.add(canopy.kernel_dict(ctx=ctx))

        result.add(
            self.surface,
            self.illumination,
            *self.measures,
            self.integrator,
            ctx=ctx,
        )

        return result
