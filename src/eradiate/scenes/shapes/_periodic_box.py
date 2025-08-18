from __future__ import annotations

import attrs

from ._cuboid import CuboidShape
from ...attrs import define, documented


@define(eq=False, slots=False)
class PeriodicBoxShape(CuboidShape):
    """
    Periodic box shape that enforces a null BSDF.

    This shape represents a cuboid with a null BSDF, which is used to define
    periodic boundaries in Monte Carlo simulations. The null BSDF ensures that
    no scattering occurs at the boundary, preserving energy conservation in
    periodic systems.

    Notes
    -----
    * The BSDF is automatically set to null and cannot be overridden.
    * This shape inherits all geometric properties from :class:`.CuboidShape`.
    """

    bsdf: None = documented(
        attrs.field(
            default=None,
            init=False,
            validator=attrs.validators.optional(
                lambda instance, attribute, value: None
            ),
        ),
        doc="BSDF attached to the shape. Always null for periodic boundaries.",
        type="None",
        init_type="Not settable",
        default="None",
    )

    def __attrs_post_init__(self):
        # Ensure BSDF is always None/null
        object.__setattr__(self, "bsdf", None)

    @property
    def template(self) -> dict:
        # Override template to explicitly set null BSDF
        result = super().template
        result["bsdf"] = {"type": "null"}
        return result
