from __future__ import annotations

import attrs
import numpy as np
import pint

from ._core import Spectrum
from ..core import KernelDict
from ..._mode import SpectralMode, supported_mode
from ...attrs import parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext
from ...radprops.rayleigh import compute_sigma_s_air
from ...units import PhysicalQuantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _eval_impl(w: pint.Quantity) -> pint.Quantity:
    return compute_sigma_s_air(wavelength=w)


def _eval_impl_ckd(w: pint.Quantity) -> pint.Quantity:
    values = _eval_impl(w)
    return np.trapz(values, w) / (w.max() - w.min())


@parse_docs
@attrs.define
class AirScatteringCoefficientSpectrum(Spectrum):
    """
    Air scattering coefficient spectrum [``air_scattering_coefficient``].

    See Also
    --------
    :func:`~eradiate.radprops.rayleigh.compute_sigma_s_air`

    Notes
    -----
    Evaluation is as follows:

    * in ``mono_*`` modes, the spectrum is evaluated at the spectral context
      wavelength;
    * in ``ckd_*`` modes, the spectrum is evaluated as the average value over
      the spectral context bin (the integral is computed using a trapezoid
      rule).
    """

    quantity: PhysicalQuantity = attrs.field(
        default=PhysicalQuantity.COLLISION_COEFFICIENT,
        init=False,
        repr=False,
    )

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return compute_sigma_s_air(wavelength=w)

    def eval_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        # Spectrum is averaged over spectral bin

        result = np.zeros((len(bindexes),))
        quantity_units = ucc.get(self.quantity)

        for i_bindex, bindex in enumerate(bindexes):
            bin = bindex.bin

            # -- Build a spectral mesh with spacing finer than 1 nm
            #    (reasonably accurate)
            wmin_m = bin.wmin.m_as(ureg.nm)
            wmax_m = bin.wmax.m_as(ureg.nm)
            w = np.linspace(wmin_m, wmax_m, 2)
            n = 10

            while True:
                if w[1] - w[0] <= 1.0:  # nm
                    break
                w = np.linspace(wmin_m, wmax_m, n + 1)
                n *= 2

            w = w * ureg.nm

            # -- Evaluate spectrum at wavelengths
            interp = self.eval_mono(w)

            # -- Average spectrum on bin extent
            integral = np.trapz(interp, w)
            result[i_bindex] = (integral / bin.width).m_as(quantity_units)

        return result * quantity_units

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        supported_mode(spectral_mode=SpectralMode.MONO | SpectralMode.CKD)

        return KernelDict(
            {
                "spectrum": {
                    "type": "uniform",
                    "value": float(
                        self.eval(ctx.spectral_ctx).m_as(
                            uck.get("collision_coefficient")
                        )
                    ),
                }
            }
        )

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError
