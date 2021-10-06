from __future__ import annotations

import attr

from ._core import PhaseFunction, phase_function_factory
from ..core import KernelDict
from ...attrs import parse_docs
from ...contexts import KernelDictContext


@phase_function_factory.register(type_id="isotropic")
@parse_docs
@attr.s
class IsotropicPhaseFunction(PhaseFunction):
    """
    Isotropic phase function [``isotropic``].

    The isotropic phase function models scattering with equal probability in
    all directions.
    """

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        return KernelDict({self.id: {"type": "isotropic"}})
