from __future__ import annotations

import datetime
import typing as t

import astropy.coordinates
import astropy.time
import astropy.units
import attr
import dateutil
import numpy as np
import pint
import xarray as xr

from ._core import Spectrum, spectrum_factory
from ..core import KernelDict
from ... import data, validators
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext
from ...units import PhysicalQuantity, to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg

# This table maps known spectra to their relative paths in the data store
_KNOWN_SPECTRA = {
    "blackbody_sun": "spectra/solar_irradiance/blackbody_sun.nc",
    "meftah_2017": "spectra/solar_irradiance/meftah_2017.nc",
    "solid_2017_mean": "spectra/solar_irradiance/solid_2017_mean.nc",
    "thuillier_2003": "spectra/solar_irradiance/thuillier_2003.nc",
    "whi_2008": "spectra/solar_irradiance/whi_2008_time_period_1.nc",
    "whi_2008_time_period_1": "spectra/solar_irradiance/whi_2008_time_period_1.nc",
    "whi_2008_time_period_2": "spectra/solar_irradiance/whi_2008_time_period_2.nc",
    "whi_2008_time_period_3": "spectra/solar_irradiance/whi_2008_time_period_3.nc",
}


def _dataset_converter(x: t.Any):
    if isinstance(x, str):
        if x in _KNOWN_SPECTRA:
            return data.open_dataset(_KNOWN_SPECTRA[x])

    return x


@spectrum_factory.register(type_id="solar_irradiance")
@parse_docs
@attr.s(frozen=True)
class SolarIrradianceSpectrum(Spectrum):
    """
    Solar irradiance spectrum [``solar_irradiance``].

    This scene element produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this element is controlled by the ``dataset`` field.

    See Also
    --------

    :ref:`Solar irradiance spectrum data guide <sec-user_guide-data-solar_irradiance>`

    Notes
    ------

    * The spectral range of the data sets shipped can vary and an attempt for
      use outside of the supported spectral range will raise a
      :class:`ValueError` upon calling :meth:`kernel_dict`.

    * The spectrum is automatically adjusted when the ``datetime`` field is set.
      Datasets without a time dimension are assumed to be normalised to a given
      Earth-Sun distance and the data is scaled based on the actual Earth-Sun
      distance for the specified date, computed using the ephemeris of
      :func:`astropy.coordinates.get_sun`.

    * The ``scale`` field can be used to apply additional arbitrary scaling.
      It is mostly used for debugging purposes.

    * The evaluation method depends on the active mode:

      * in ``mono_*`` modes, the spectrum is evaluated at the spectral context
        wavelength;
      * in ``ckd_*`` modes, the spectrum is evaluated as the average value over
        the spectral context bin (the integral is computed using a trapezoid
        rule).

    * The produced kernel dictionary automatically adjusts its irradiance units
      depending on the selected kernel default units.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    quantity: PhysicalQuantity = attr.ib(
        default=PhysicalQuantity.IRRADIANCE, init=False, repr=False
    )

    dataset: xr.Dataset = documented(
        attr.ib(
            default="thuillier_2003",
            converter=_dataset_converter,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Solar spectrum dataset. Optionally, a keyword can be passed to "
        "fetch a registered spectrum from the data store. See "
        ":ref:`sec-user_guide-data-solar_irradiance` for the list of spectra "
        "shipped with Eradiate.",
        type="Dataset",
        init_type="Dataset or str, optional",
        default='"thuillier_2003"',
    )

    scale: float = documented(
        attr.ib(default=1.0, converter=float, validator=validators.is_positive),
        doc="Arbitrary scaling factor.",
        type="float or datetime",
        init_type="float or datetime or str",
        default="1.0",
    )

    datetime: t.Optional[datetime.datetime] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(dateutil.parser.parse),
        ),
        type="datetime or None",
        init_type="datetime or str, optional",
        doc="Date for which the spectrum is to be evaluated. An ISO "
        "string can be passed and will be interpreted by "
        ":meth:`dateutil.parser.parse`.",
    )

    def _scale_earth_sun_distance(self) -> float:
        """
        Compute scaling factor applied to the irradiance spectrum based on the
        Earth-Sun distance.
        """
        # Note: We assume that the loaded dataset is for a reference
        # Earth-Sun distance of 1 AU
        if self.datetime is None:
            return 1.0

        else:
            return (
                float(
                    astropy.coordinates.get_sun(
                        astropy.time.Time(self.datetime)
                    ).distance
                    / astropy.units.au
                )
                ** 2
            )

    def eval_mono(self, w: pint.Quantity) -> pint.Quantity:
        # Inherit docstring

        w_units = ureg(self.dataset.ssi.w.attrs["units"])
        irradiance = to_quantity(
            self.dataset.ssi.interp(w=w.m_as(w_units), method="linear")
        )

        # Raise if out of bounds or ill-formed dataset
        if np.any(np.isnan(irradiance.magnitude)):
            raise ValueError("dataset interpolation returned nan")

        return irradiance * self.scale * self._scale_earth_sun_distance()

    def eval_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        # Inherit docstring
        # Note: Spectrum is averaged over the spectral bin

        result = np.zeros((len(bindexes),))
        wavelength_units = ucc.get("wavelength")
        quantity_units = ucc.get(self.quantity)

        for i_bindex, bindex in enumerate(bindexes):
            bin = bindex.bin

            wmin_m = bin.wmin.m_as(wavelength_units)
            wmax_m = bin.wmax.m_as(wavelength_units)

            # -- Collect relevant spectral coordinate values
            w_m = ureg.convert(
                self.dataset.ssi.w.values,
                self.dataset.ssi.w.attrs["units"],
                wavelength_units,
            )
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

            # -- Evaluate spectrum at wavelengths
            interp = self.eval_mono(w)

            # -- Average spectrum on bin extent
            integral = np.trapz(interp, w)
            result[i_bindex] = (integral / bin.width).m_as(quantity_units)

        return result * quantity_units

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Apply scaling, build kernel dict
        value = float(self.eval(ctx.spectral_ctx).m_as(uck.get("irradiance")))
        return KernelDict(
            {
                "spectrum": {
                    "type": "uniform",
                    "value": value,
                }
            }
        )

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError
