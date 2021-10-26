from __future__ import annotations

import typing as t
from abc import ABC

import attr
import pinttr

from ...attrs import documented, parse_docs
from ...scenes.core import SceneElement
from ...scenes.illumination import (
    ConstantIllumination,
    DirectionalIllumination,
    illumination_factory,
)
from ...scenes.integrators import Integrator, PathIntegrator, integrator_factory
from ...scenes.measure import Measure, measure_factory
from ...units import unit_registry as ureg


@parse_docs
@attr.s
class Scene(SceneElement, ABC):
    """
    Abstract class common to all scenes.
    """

    illumination: DirectionalIllumination = documented(
        attr.ib(
            factory=DirectionalIllumination,
            converter=illumination_factory.convert,
            validator=attr.validators.instance_of(
                (DirectionalIllumination, ConstantIllumination)
            ),
        ),
        doc="Illumination specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`IlluminationFactory.convert() <.IlluminationFactory.convert>`.",
        type=":class:`.DirectionalIllumination`",
        init_type=":class:`.DirectionalIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
    )

    measures: t.List[Measure] = documented(
        attr.ib(
            factory=lambda: [DistantRadianceMeasure()],
            converter=lambda value: [
                measure_factory.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [measure_factory.convert(value)],
        ),
        doc="List of measure specifications. The passed list may contain "
        "dictionaries, which will be interpreted by "
        ":data:`.measure_factory`. "
        "Optionally, a single :class:`.Measure` or dictionary specification "
        "may be passed and will automatically be wrapped into a list.\n"
        "\n"
        "Allowed value types: :class:`DistantMeasure`, "
        ":class:`DistantReflectanceMeasure`.\n"
        "\n"
        ".. note:: The target zone will be overridden using canopy "
        "parameters if unset. If no canopy is specified, surface size "
        "parameters will be used.",
        init_type="list of (Measure|dict) or :class:`.Measure` or dict",
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
            converter=integrator_factory.convert,
            validator=attr.validators.instance_of(Integrator),
        ),
        doc="Monte Carlo integration algorithm specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by "
        ":meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`.",
        type=":class:`.Integrator`",
        init_type=":class:`.Integrator` or dict",
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    def update(self):
        pass

    def __attrs_post_init__(self):
        self.update()

    @classmethod
    def from_dict(cls, d) -> Scene:
        """
        Create from dictionary. This constructor implements a unit processing
        step based on :func:`pinttr.interpret_units`.

        Parameters
        ----------
        d : dict
            Dictionary.

        Returns
        -------
        :class:`Scene`
        """
        d_copy = pinttr.interpret_units(d, ureg=ureg)
        return cls(**d_copy)
