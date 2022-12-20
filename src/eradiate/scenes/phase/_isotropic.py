from __future__ import annotations

import attrs

from ._core import PhaseFunctionNode
from ...attrs import parse_docs


@parse_docs
@attrs.define(eq=False, slots=False)
class IsotropicPhaseFunction(PhaseFunctionNode):
    """
    Isotropic phase function [``isotropic``].

    The isotropic phase function models scattering with equal probability in
    all directions.
    """

    @property
    def template(self) -> dict:
        return {"type": "isotropic"}
