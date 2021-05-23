from abc import ABC
from typing import List

import attr
import pinttr

from ..._attrs import documented, parse_docs
from ...scenes.core import SceneElement
from ...scenes.illumination import (
    DirectionalIllumination,
    Illumination,
    IlluminationFactory,
)
from ...scenes.integrators import Integrator, IntegratorFactory, PathIntegrator
from ...scenes.measure import DistantMeasure, Measure, MeasureFactory


@parse_docs
@attr.s
class Scene(SceneElement, ABC):
    """
    Abstract class common to all scenes.
    """

    illumination: Illumination = documented(
        attr.ib(
            factory=DirectionalIllumination,
            converter=IlluminationFactory.convert,
            validator=attr.validators.instance_of(DirectionalIllumination),
        ),
        doc="Illumination specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`IlluminationFactory.convert() <.IlluminationFactory.convert>`.",
        type=":class:`.DirectionalIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
    )

    measures: List[Measure] = documented(
        attr.ib(
            factory=lambda: [DistantMeasure()],
            converter=lambda value: [
                MeasureFactory.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [MeasureFactory.convert(value)],
        ),
        doc="List of measure specifications. The passed list may contain "
        "dictionaries, which will be interpreted by "
        ":meth:`MeasureFactory.convert() <.MeasureFactory.convert>`. "
        "Optionally, a single :class:`.Measure` or dictionary specification "
        "may be passed and will automatically be wrapped into a list.\n"
        "\n"
        "Allowed value types: :class:`DistantMeasure`.\n"
        "\n"
        ".. note:: The target zone will be overridden using canopy "
        "parameters if unset. If no canopy is specified, surface size "
        "parameters will be used.",
        type="list[:class:`.Measure`] or list[dict] or :class:`.Measure` or dict",
        default=":class:`DistantMeasure() <.DistantMeasure>`",
    )

    @measures.validator
    def _measures_validator(self, attribute, value):
        for element in value:
            # Check measure type
            if not isinstance(element, DistantMeasure):
                raise TypeError(
                    f"while validating {attribute.name}: must be a list of "
                    f"objects of one of the following types: "
                    f"(DistantMeasure)"
                )

    integrator: Integrator = documented(
        attr.ib(
            factory=PathIntegrator,
            converter=IntegratorFactory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc="Monte Carlo integration algorithm specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`.",
        type=":class:`.Integrator` or dict",
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    def update(self):
        pass

    def __attrs_post_init__(self):
        self.update()
