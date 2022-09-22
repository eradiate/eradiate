from __future__ import annotations

import attrs

from ._core import PhaseFunction
from ..core import KernelDict
from ...attrs import parse_docs
from ...contexts import KernelDictContext


@parse_docs
@attrs.define
class IsotropicPhaseFunction(PhaseFunction):
    """
    Isotropic phase function [``isotropic``].

    The isotropic phase function models scattering with equal probability in
    all directions.
    """

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        return KernelDict({self.id: {"type": "isotropic"}})
