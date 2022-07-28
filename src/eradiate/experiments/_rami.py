import typing as t

import attr

from ._core import EarthObservationExperiment, Experiment
from .. import validators
from ..attrs import documented, get_doc, parse_docs
from ..contexts import KernelDictContext
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.bsdfs import BSDF, LambertianBSDF, bsdf_factory
from ..scenes.core import KernelDict
from ..scenes.integrators import Integrator, PathIntegrator, integrator_factory
from ..scenes.measure import Measure
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, surface_factory


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
@attr.s
class RamiExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and no atmosphere.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.

    Notes
    -----
    A post-initialisation step will constrain the measure setup if a
    distant measure is used and no target is defined:

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

    surface: t.Union[None, BasicSurface] = documented(
        attr.ib(
            factory=lambda: LambertianBSDF(),
            converter=attr.converters.optional(_surface_converter),
            validator=attr.validators.optional(
                attr.validators.instance_of(BasicSurface)
            ),
        ),
        doc="Surface specification. If set to ``None``, no surface will be "
        "added. This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.surface_factory` and :data:`.bsdf_factory`. "
        "**If relevant, the surface size will be adjusted automatically upon "
        "kernel. dictionary generation.**",
        type=".BasicSurface or None",
        init_type=".BasicSurface or .BSDF or dict, optional",
        default=":class:`BasicSurface(bsdf=LambertianBSDF()) <.BasicSurface>`",
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

    def __attrs_post_init__(self):
        self._normalize_measures()

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = KernelDict()

        # Process canopy
        if self.canopy is not None:
            scene_width = max(self.canopy.size[:2])

            if self.padding > 0:  # We must add extra instances if padding is requested
                scene_width *= 2.0 * self.padding + 1.0
                canopy = self.canopy.padded_copy(self.padding)
            else:
                canopy = self.canopy

            result.add(canopy, ctx=ctx)

            # Surface size always matches canopy size
            if self.surface is not None:
                surface = attr.evolve(
                    self.surface,
                    shape=RectangleShape(center=[0, 0, 0], edges=scene_width),
                )

            else:
                surface = None

        else:
            surface = attr.evolve(self.surface)

        # Process surface
        if surface is not None:
            result.add(surface, ctx=ctx)

        # Process measures
        result.add(*self.measures, ctx=ctx)

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
            # Override ray target location if relevant
            if (
                measure.is_distant() and measure.target is None
            ):  # No target specified: add one
                if self.canopy is None:  # No canopy: target origin point
                    measure.target = {"type": "point", "xyz": [0, 0, 0]}
                else:  # Canopy: target top of canopy
                    measure.target = {
                        "type": "rectangle",
                        "xmin": -0.5 * self.canopy.size[0],
                        "xmax": 0.5 * self.canopy.size[0],
                        "ymin": -0.5 * self.canopy.size[1],
                        "ymax": 0.5 * self.canopy.size[1],
                        "z": self.canopy.size[2],
                    }

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        result = super(RamiExperiment, self)._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-canopy simulation results"

        return result
