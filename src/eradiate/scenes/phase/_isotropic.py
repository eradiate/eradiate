from ._core import PhaseFunction
from ...attrs import define


@define(eq=False, slots=False)
class IsotropicPhaseFunction(PhaseFunction):
    """
    Isotropic phase function [``isotropic``].

    The isotropic phase function models scattering with equal probability in
    all directions.
    """

    @property
    def template(self) -> dict:
        return {"type": "isotropic"}
