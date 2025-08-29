from __future__ import annotations

import attrs

from ._core import Integrator
from ..core import BoundingBox
from ...attrs import define, documented
from ...units import unit_context_kernel as uck


@define(eq=False, slots=False)
class PAccumulatorIntegrator(Integrator):
    """
    A thin interface to the particle accumualtor kernel plugin [``paccumulator``].

    This integrator samples paths using random wolks starting from the emitter.
    It supports multiple scattering and accounts for volume interactions.
    The values it accumulate depend on the sensor.

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

    periodic_box: BoundingBox | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(BoundingBox.convert),
        ),
        doc="Bounding box of the periodic boundary. Rays exiting one face "
        "of the boundary will enter back from the opposing face. Note that "
        "rays must originate from inside the periodic box when specified. "
        "See the family of periodic emitters e.g. :class:`.DirectionalPeriodicIllumination`",
        type=":class:`.BoundingBox` or None",
        init_type=":class:`.BoundingBox`, dict, tuple, or array-like, optional",
        default=None,
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

        if self.periodic_box is not None:
            result["pbox_min"] = self.periodic_box.min.m_as(uck.get("length"))
            result["pbox_max"] = self.periodic_box.max.m_as(uck.get("length"))
        else:
            raise ValueError(
                "Either 'pbox_min'/'pbox_max' or 'periodic_box' must be specified."
            )

        return result
