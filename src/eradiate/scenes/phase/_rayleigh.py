from __future__ import annotations

import attrs

import eradiate

from ._core import PhaseFunction
from ...attrs import define, documented


@define(eq=False, slots=False)
class RayleighPhaseFunction(PhaseFunction):
    """
    Rayleigh phase function [``rayleigh``].

    The Rayleigh phase function models scattering by particles with a
    characteristic size much smaller than the considered radiation wavelength.
    It is typically used to represent scattering by gas molecules in the
    atmosphere.
    """

    depolarization: float | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(float),
        ),
        doc="The ratio of intensities parallel and perpendicular to the"
        "plane of scattering for light scattered at 90 deg. Only relevant"
        "when using a polarization mode.",
        type="float or None",
        init_type="float, optional",
        default="None",
    )

    @property
    def template(self) -> dict:
        phase_function = "rayleigh"
        if eradiate.mode().is_polarized:
            phase_function = "rayleigh_polarized"

        result = {"type": phase_function}
        if self.depolarization is not None:
            result["depolarization"] = self.depolarization
        return result
