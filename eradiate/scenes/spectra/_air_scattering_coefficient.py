import attr

import eradiate

from ._core import Spectrum, SpectrumFactory
from ..._attrs import parse_docs
from ..._units import PhysicalQuantity
from ..._units import unit_context_kernel as uck
from ...exceptions import ModeError
from ...radprops.rayleigh import compute_sigma_s_air


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

    def eval(self, spectral_ctx=None):
        if eradiate.mode().is_monochromatic():
            return compute_sigma_s_air(wavelength=spectral_ctx.wavelength)
        else:
            raise ModeError(f"unsupported mode {eradiate.mode().id}")

    def kernel_dict(self, ctx=None):
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
