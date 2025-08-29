from __future__ import annotations

import attrs
import pint
import pinttrs

from ._core import Integrator
from ...attrs import define, documented
from ...units import unit_context_kernel as ucc
from ...units import unit_context_kernel as uck
from ...validators import is_vector3, on_quantity


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

    pbox_min: pint.Quantity | None = documented(
        pinttrs.field(
            default=None,
            validator=attrs.validators.optional(
                (pinttrs.validators.has_compatible_units, on_quantity(is_vector3))
            ),
            units=ucc.deferred("length"),
        ),
        doc="Minimum point of the periodic bounding box. Must be used together with "
        "`pbox_max`.\n\n"
        "Unit-enabled field (default units: ucc['length']).",
        type="quantity or None",
        init_type="quantity or array-like, optional",
        default="None",
    )

    pbox_max: pint.Quantity | None = documented(
        pinttrs.field(
            default=None,
            validator=attrs.validators.optional(
                (pinttrs.validators.has_compatible_units, on_quantity(is_vector3))
            ),
            units=ucc.deferred("length"),
        ),
        doc="Maximum point of the periodic bounding box. Must be used together with "
        "`pbox_min`.\n\n"
        "Unit-enabled field (default units: ucc['length']).",
        type="quantity or None",
        init_type="quantity or array-like, optional",
        default="None",
    )

    @property
    def kernel_type(self) -> str:
        return "paccumulator"

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

        if self.pbox_min is not None and self.pbox_max is not None:
            result["pbox_min"] = self.pbox_min.m_as(uck.get("length"))
            result["pbox_max"] = self.pbox_max.m_as(uck.get("length"))
        else:
            raise ValueError(
                "Either 'pbox_min'/'pbox_max' or 'periodic_box' must be specified."
            )

        return result
