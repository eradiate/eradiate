from __future__ import annotations

import typing as t

import attrs
import pint

from ._core import Spectrum
from ..core import KernelDict
from ..._mode import SpectralMode, supported_mode
from ...attrs import parse_docs
from ...contexts import KernelDictContext
from ...radprops.rayleigh import compute_sigma_s_air
from ...units import PhysicalQuantity
from ...units import unit_context_kernel as uck


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

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_mono(w)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        supported_mode(spectral_mode=SpectralMode.MONO | SpectralMode.CKD)
        
        kernel_units = uck.get("collision_coefficient")
        value = float(self.eval(ctx.spectral_index).m_as(kernel_units))
        return KernelDict({"spectrum": {"type": "uniform", "value": value}})

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError
