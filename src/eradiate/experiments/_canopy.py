from __future__ import annotations

import typing as t

import attrs

from ._core import EarthObservationExperiment
from ._helpers import surface_converter
from .. import validators
from ..attrs import AUTO, define, documented
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import SceneElement
from ..scenes.integrators import VolPathIntegrator
from ..scenes.measure import Measure
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface


@define
class CanopyExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and no atmosphere.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.

    Notes
    -----
    A post-initialization step will constrain the measure setup if a
    distant measure is used and no target is defined:

    * if a canopy is defined, the target will be set to the top of the canopy
      unit cell (*i.e.* without its padding);
    * if no canopy is defined, the target will be set to [0, 0, 0].
    """

    canopy: Canopy | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(biosphere_factory.convert),
            validator=attrs.validators.optional(attrs.validators.instance_of(Canopy)),
        ),
        doc="Canopy specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.biosphere_factory`.",
        type=":class:`.Canopy` or None",
        init_type=":class:`.Canopy` or dict or None",
        default="None",
    )

    padding: int = documented(
        attrs.field(default=0, converter=int, validator=validators.is_positive),
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

    surface: None | BasicSurface = documented(
        attrs.field(
            factory=lambda: LambertianBSDF(),
            converter=attrs.converters.optional(surface_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(BasicSurface)
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

    def __attrs_post_init__(self):
        self._normalize_spectral()
        self._normalize_measures()
        self._normalize_integrator()

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

    def _normalize_integrator(self) -> None:
        """
        Ensures that the integrator is compatible with the atmosphere and geometry.
        """

        if self.integrator is AUTO:
            self.integrator = VolPathIntegrator()

    def _dataset_metadata(self, measure: Measure) -> dict[str, str]:
        result = super()._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-canopy simulation results"

        return result

    def _context_kwargs(self) -> dict[str, t.Any]:
        return {}

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        # Inherit docstring

        objects = {}

        # Process canopy and surface
        if self.canopy is not None:
            scene_width = max(self.canopy.size[:2])

            if self.padding > 0:  # Add extra instances if padding is requested
                scene_width *= 2.0 * self.padding + 1.0
                objects["canopy"] = self.canopy.padded_copy(self.padding)
            else:
                objects["canopy"] = self.canopy

            if self.surface is not None:  # Adjust surface to match canopy
                objects["surface"] = attrs.evolve(
                    self.surface,
                    shape=RectangleShape(center=[0, 0, 0], edges=scene_width),
                )

        else:
            if self.surface is not None:  # Leave surface unchanged
                objects["surface"] = self.surface

        objects.update(
            {
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )

        return objects
