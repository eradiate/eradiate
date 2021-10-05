from __future__ import annotations

import attr
import numpy as np
import pint

import eradiate

from ._core import Spectrum, spectrum_factory
from ..core import KernelDict
from ..._mode import ModeFlags
from ...attrs import parse_docs
from ...ckd import Bin, Bindex
from ...contexts import KernelDictContext
from ...exceptions import UnsupportedModeError
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


@spectrum_factory.register(type_id="air_scattering_coefficient")
@parse_docs
@attr.s
class AirScatteringCoefficientSpectrum(Spectrum):
    """
    Air scattering coefficient spectrum.

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

    quantity: PhysicalQuantity = attr.ib(
        default=PhysicalQuantity.COLLISION_COEFFICIENT,
        init=False,
        repr=False,
    )

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        return compute_sigma_s_air(wavelength=w)

    def eval_rgb(self) -> pint.Quantity:
        quantity_units = ucc.get(self.quantity)
        return (
            np.array(
                [
                    compute_sigma_s_air(wavelength=wavelength).m_as(quantity_units)
                    for wavelength in [600, 500, 400] * ureg.nm
                ]
            )
            * quantity_units
        )

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
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
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
        elif eradiate.mode().has_flags(ModeFlags.ANY_RGB):
            from mitsuba.core import ScalarColor3f

            return KernelDict(
                {
                    "spectrum": {
                        "type": "srgb",
                        "color": list(
                            self.eval(ctx.spectral_ctx).m_as(
                                uck.get("collision_coefficient")
                            )
                        ),
                    }
                }
            )
        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd", "rgb"))
