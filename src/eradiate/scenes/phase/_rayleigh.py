import attrs

import eradiate

from ._core import PhaseFunction
from ...attrs import parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class RayleighPhaseFunction(PhaseFunction):
    """
    Rayleigh phase function [``rayleigh``].

    The Rayleigh phase function models scattering by particles with a
    characteristic size much smaller than the considered radiation wavelength.
    It is typically used to represent scattering by gas molecules in the
    atmosphere.
    """

    @property
    def template(self) -> dict:
        phase_function = "rayleigh"
        if eradiate.mode().is_polarized:
            phase_function = "rayleigh_polarized"

        return {"type": phase_function}
