from __future__ import annotations

import attrs

import eradiate

from ._core import Integrator
from ..core import NodeSceneElement, Ref
from ...attrs import define, documented


@define(eq=False, slots=False)
class PAccumulatorIntegrator(Integrator):
    """
    Base class for integrator elements wrapping kernel classes
    deriving from
    :class:`mitsuba.MonteCarloIntegrator`.

    .. warning:: This class should not be instantiated.
    """

    min_depth: int | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Minimum path depth in the generated measure data (where -1 "
        "corresponds to ∞). A value of 1 will display only visible emitters. 2 "
        "computes only direct illumination (no multiple scattering), etc. If "
        "unset, the kernel default value (-1) is used.",
        type="int or None",
        init_type="int, optional",
    )


    max_depth: int | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Longest path depth in the generated measure data (where -1 "
        "corresponds to ∞). A value of 1 will display only visible emitters. 2 "
        "computes only direct illumination (no multiple scattering), etc. If "
        "unset, the kernel default value (-1) is used.",
        type="int or None",
        init_type="int, optional",
    )

    rr_depth: int | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(int)),
        doc="Minimum path depth after which the implementation starts applying "
        "the Russian roulette path termination criterion. If unset, the kernel "
        "default value (5) is used.",
        type="int or None",
        init_type="int, optional",
    )

    periodic_box: Ref | None = documented( 
        attrs.field(
            default=None,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(Ref)
            ),
        ),
        doc="Periodic box shape which defines the boundary that conserve energy. "
        "If unset, the kernel defaults to no periodicity. "
        "Important: this must be a reference to a shape in the scene. It also "
        "must have a null bsdf.",
        init_type="Ref, optional",
    )

    @property
    def kernel_type(self) -> str:
        return "paccumulator"

    @property
    def objects(self) -> dict[str, NodeSceneElement] | None:
        # Inherit docstring
        if self.periodic_box is None:
            return None
        else:
            return {"periodic_box": self.periodic_box}

    @property
    def template(self) -> dict:
        result = {"type": self.kernel_type}

        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.min_depth is not None:
            result["min_depth"] = self.min_depth
        if self.max_depth is not None:
            result["max_depth"] = self.max_depth
        if self.rr_depth is not None:
            result["rr_depth"] = self.rr_depth
        # if self.periodic_box is not None:
        #     result["periodic_box"] = {"type":"ref", "id": self.periodic_box.id()}

        return result


