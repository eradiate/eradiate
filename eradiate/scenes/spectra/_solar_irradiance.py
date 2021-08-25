from typing import MutableMapping, Optional

import attr
import numpy as np
import pint
import xarray as xr

import eradiate

from ._core import Spectrum, spectrum_factory
from ... import data
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...units import PhysicalQuantity, to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import is_positive


def _eval_impl(ssi: xr.DataArray, w: pint.Quantity) -> pint.Quantity:
    w_units = ureg[ssi.w.attrs["units"]]
    irradiance = to_quantity(ssi.interp(w=w.m_as(w_units), method="linear"))

    # Raise if out of bounds or ill-formed dataset
    if np.any(np.isnan(irradiance.magnitude)):
        raise ValueError("dataset interpolation returned nan")

    return irradiance


def _eval_impl_ckd(ssi: xr.DataArray, w: pint.Quantity) -> pint.Quantity:
    interp = _eval_impl(ssi, w)
    return np.trapz(interp, w) / (w.max() - w.min())


@spectrum_factory.register(type_id="solar_irradiance")
@parse_docs
@attr.s(frozen=True)
class SolarIrradianceSpectrum(Spectrum):
    """
    Solar irradiance spectrum scene element [``solar_irradiance``].

    This scene element produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this element is controlled by the ``dataset`` attribute (see
    :mod:`eradiate.data.solar_irradiance_spectra` for available data sets).

    The spectral range of the data sets shipped can vary and an attempt for use
    outside of the supported spectral range will raise a :class:`ValueError`
    upon calling :meth:`kernel_dict`.

    The generated kernel dictionary varies based on the selected mode of
    operation. The ``scale`` parameter can be used to adjust the value based on
    unit conversion or to account for variations of the Sun-planet distance.

    The produced kernel dictionary automatically adjusts its irradiance units
    depending on the selected kernel default units.

    Evaluation is as follows:

    * in ``mono_*`` modes, the spectrum is evaluated at the spectral context
      wavelength;
    * in ``ckd_*`` modes, the spectrum is evaluated as the average value over
      the spectral context bin (the integral is computed using a trapezoid
      rule).
    """

    quantity = attr.ib(default=PhysicalQuantity.IRRADIANCE, init=False, repr=False)

    dataset = documented(
        attr.ib(
            default="thuillier_2003",
            validator=attr.validators.instance_of(str),
        ),
        doc="Dataset identifier. Allowed values: see "
        ":attr:`solar irradiance dataset documentation <eradiate.data.solar_irradiance_spectra>`. "
        'Default: ``"thuillier_2003"``. ',
        type="str",
    )

    scale = documented(
        attr.ib(
            default=1.0,
            converter=float,
            validator=is_positive,
        ),
        doc="Scaling factor. Default: 1.",
        type="float",
    )

    @dataset.validator
    def _dataset_validator(self, attribute, value):
        if value not in data.registered("solar_irradiance_spectrum"):
            raise ValueError(
                f"while setting {attribute.name}: '{value}' not in "
                f"list of supported solar irradiance spectra "
                f"{data.registered('solar_irradiance_spectrum')}"
            )

    data = attr.ib(init=False, repr=False)

    @data.default
    def _data_factory(self):
        # Load dataset
        try:
            return data.open("solar_irradiance_spectrum", self.dataset)
        except KeyError:
            raise ValueError(f"unknown dataset {self.dataset}")

    def eval(self, spectral_ctx: SpectralContext = None) -> pint.Quantity:
        if self.dataset == "solid_2017":
            raise NotImplementedError(
                "Solar irradiance spectrum datasets with a non-empty time "
                "coordinate are not supported yet."
            )
        # TODO: add support to solar irradiance spectrum datasets with a
        #  non-empty time coordinate

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return _eval_impl(self.data.ssi, spectral_ctx.wavelength)

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            # Average irradiance over spectral bin
            wavelength_units = ucc.get("wavelength")
            wmin_m = spectral_ctx.bin.wmin.m_as(wavelength_units)
            wmax_m = spectral_ctx.bin.wmax.m_as(wavelength_units)

            # -- Collect relevant spectral coordinate values
            w_m = to_quantity(self.data.ssi.w).m_as(wavelength_units)
            w = (
                np.hstack(
                    (
                        [wmin_m],
                        w_m[np.where(np.logical_and(wmin_m < w_m, w_m < wmax_m))[0]],
                        [wmax_m],
                    )
                )
                * wavelength_units
            )

            # Interpolate at collected wavelengths and compute average on bin extent
            return _eval_impl_ckd(self.data.ssi, w)

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        # Apply scaling, build kernel dict
        return {
            "spectrum": {
                "type": "uniform",
                "value": self.eval(ctx.spectral_ctx).m_as(uck.get("irradiance"))
                * self.scale,
            }
        }
