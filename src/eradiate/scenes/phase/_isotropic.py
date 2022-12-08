from __future__ import annotations

import attrs

from ._core import PhaseFunction
from ..core import NodeSceneElement
from ...attrs import parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class IsotropicPhaseFunction(PhaseFunction, NodeSceneElement):
    """
    Isotropic phase function [``isotropic``].

    The isotropic phase function models scattering with equal probability in
    all directions.
    """

    @property
    def kernel_type(self) -> str:
        return "isotropic"
