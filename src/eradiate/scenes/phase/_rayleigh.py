import attrs

from ._core import PhaseFunctionNode
from ...attrs import parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class RayleighPhaseFunction(PhaseFunctionNode):
    """
    Rayleigh phase function [``rayleigh``].

    The Rayleigh phase function models scattering by particles with a
    characteristic size much smaller than the considered radiation wavelength.
    It is typically used to represent scattering by gas molecules in the
    atmosphere.
    """

    @property
    def template(self) -> dict:
        return {"type": "rayleigh"}
