from typing import MutableMapping, Optional

import attr

from ._core import PhaseFunction, phase_function_factory
from ...attrs import parse_docs
from ...contexts import KernelDictContext


@phase_function_factory.register(type_id="rayleigh")
@parse_docs
@attr.s
class RayleighPhaseFunction(PhaseFunction):
    """
    The Rayleigh phase function models scattering by particles with a
    characteristic size much smaller than the considered radiation wavelength.
    It is typically used to represent scattering by gas molecules in the
    atmosphere.
    """

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        return {self.id: {"type": "rayleigh"}}
