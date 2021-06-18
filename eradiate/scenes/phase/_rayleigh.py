from typing import MutableMapping, Optional

import attr

from ._core import PhaseFunction, PhaseFunctionFactory
from ..._attrs import parse_docs
from ...contexts import KernelDictContext


@PhaseFunctionFactory.register("rayleigh")
@parse_docs
@attr.s
class RayleighPhaseFunction(PhaseFunction):
    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        return {self.id: {"type": "rayleigh"}}
