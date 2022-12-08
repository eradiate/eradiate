from __future__ import annotations

import attrs

from ._core import PhaseFunction
from ..core import NodeSceneElement
from ...attrs import parse_docs
from ...contexts import KernelDictContext


@parse_docs
@attrs.define(eq=False, slots=False)
class RayleighPhaseFunction(PhaseFunction, NodeSceneElement):
    """
    Rayleigh phase function [``rayleigh``].

    The Rayleigh phase function models scattering by particles with a
    characteristic size much smaller than the considered radiation wavelength.
    It is typically used to represent scattering by gas molecules in the
    atmosphere.
    """

    @property
    def kernel_type(self) -> str:
        return "rayleigh"
