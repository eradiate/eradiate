from typing import MutableMapping, Optional

import attr
import pint

import eradiate

from ._core import Spectrum, SpectrumFactory
from ...attrs import parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import ModeError
from ...radprops.rayleigh import compute_sigma_s_air
from ...units import PhysicalQuantity
from ...units import unit_context_kernel as uck


@SpectrumFactory.register("air_scattering_coefficient")
@parse_docs
@attr.s(frozen=True)
class AirScatteringCoefficientSpectrum(Spectrum):
    """
    Air scattering coefficient spectrum.

    .. seealso:: :func:`~eradiate.radprops.rayleigh.compute_sigma_s_air`
    """

    quantity = attr.ib(
        default=PhysicalQuantity.COLLISION_COEFFICIENT, init=False, repr=False
    )

    def eval(self, spectral_ctx: SpectralContext = None) -> pint.Quantity:
        if eradiate.mode().is_monochromatic():
            return compute_sigma_s_air(wavelength=spectral_ctx.wavelength)
        else:
            raise ModeError(f"unsupported mode {eradiate.mode().id}")

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if eradiate.mode().is_monochromatic():
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": self.eval(ctx.spectral_ctx).m_as(
                        uck.get("collision_coefficient")
                    ),
                }
            }

        raise ModeError(f"unsupported mode '{eradiate.mode().id}'")
