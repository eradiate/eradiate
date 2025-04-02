from __future__ import annotations

import attrs
import pint

from ._core import Spectrum
from ...attrs import define
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...radprops.rayleigh import compute_sigma_s_air
from ...units import PhysicalQuantity
from ...units import unit_context_kernel as uck


@define(eq=False, slots=False)
class AirScatteringCoefficientSpectrum(Spectrum):
    """
    Air scattering coefficient spectrum [``air_scattering_coefficient``].

    See Also
    --------
    :func:`~eradiate.radprops.rayleigh.compute_sigma_s_air`
    """

    quantity: PhysicalQuantity = attrs.field(
        default=PhysicalQuantity.COLLISION_COEFFICIENT,
        init=False,
        repr=False,
    )

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        # Inherit docstring
        return compute_sigma_s_air(wavelength=w)

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        # Inherit docstring
        return self.eval_mono(w)

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError

    @property
    def template(self) -> dict:
        # Inherit docstring

        return {
            "type": "uniform",
            "value": DictParameter(
                func=lambda ctx: float(
                    self.eval(ctx.si).m_as(uck.get("collision_coefficient"))
                )
            ),
        }

    @property
    def params(self) -> dict[str, SceneParameter]:
        # Inherit docstring

        return {
            "value": SceneParameter(
                func=lambda ctx: float(
                    self.eval(ctx.si).m_as(uck.get("collision_coefficient"))
                ),
                flags=KernelSceneParameterFlags.SPECTRAL,
            )
        }
