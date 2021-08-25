from typing import MutableMapping, Optional

import attr
import numpy as np
import pint

import eradiate

from ._core import Spectrum, spectrum_factory
from ..._mode import ModeFlags
from ...attrs import parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...radprops.rayleigh import compute_sigma_s_air
from ...units import PhysicalQuantity
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _eval_impl(w: pint.Quantity) -> pint.Quantity:
    return compute_sigma_s_air(wavelength=w)


def _eval_impl_ckd(w: pint.Quantity) -> pint.Quantity:
    values = _eval_impl(w)
    return np.trapz(values, w) / (w.max() - w.min())


@spectrum_factory.register(type_id="air_scattering_coefficient")
@parse_docs
@attr.s
class AirScatteringCoefficientSpectrum(Spectrum):
    """
    Air scattering coefficient spectrum.

    .. seealso:: :func:`~eradiate.radprops.rayleigh.compute_sigma_s_air`

    Evaluation is as follows:

    * in ``mono_*`` modes, the spectrum is evaluated at the spectral context
      wavelength;
    * in ``ckd_*`` modes, the spectrum is evaluated as the average value over
      the spectral context bin (the integral is computed using a trapezoid
      rule).
    """

    quantity = attr.ib(
        default=PhysicalQuantity.COLLISION_COEFFICIENT,
        init=False,
        repr=False,
    )

    def eval(self, spectral_ctx: SpectralContext = None) -> pint.Quantity:
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return _eval_impl(spectral_ctx.wavelength)

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            # Build a spectral mesh with spacing finer than 1 nm (reasonably accurate)
            wmin_m = spectral_ctx.bin.wmin.m_as(ureg.nm)
            wmax_m = spectral_ctx.bin.wmax.m_as(ureg.nm)
            w = np.linspace(wmin_m, wmax_m, 2)
            n = 10

            while True:
                if w[1] - w[0] <= 1.0:  # nm
                    break
                w = np.linspace(wmin_m, wmax_m, n + 1)
                n *= 2

            return _eval_impl_ckd(w * ureg.nm)

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
            return {
                "spectrum": {
                    "type": "uniform",
                    "value": self.eval(ctx.spectral_ctx).m_as(
                        uck.get("collision_coefficient")
                    ),
                }
            }

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))
