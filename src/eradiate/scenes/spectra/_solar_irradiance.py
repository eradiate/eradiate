from __future__ import annotations

import datetime
import importlib.util
import warnings
from typing import Any

import attrs
import numpy as np
import pint
import xarray as xr

from ._core import Spectrum
from ... import converters, validators
from ...attrs import define, documented
from ...exceptions import DataError
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...units import PhysicalQuantity, to_quantity
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import summary_repr


def _datetime_converter(x: Any):
    if x is not None:
        try:
            import dateutil
        except ModuleNotFoundError:
            warnings.warn(
                "To use the date-based Solar irradiance scaling feature, you "
                "must install dateutil.\n"
                "See instructions on https://dateutil.readthedocs.io/#installation."
            )
            raise

        try:
            importlib.import_module("astropy")
        except ModuleNotFoundError:
            warnings.warn(
                "To use the date-based Solar irradiance scaling feature, you "
                "must install astropy.\n"
                "See instructions on https://www.astropy.org/."
            )
            raise

        return dateutil.parser.parse(x)


def _dataset_converter(value: Any):
    if isinstance(value, xr.Dataset):
        return value

    try:
        result = converters.resolve_keyword(lambda x: f"solar_irradiance/{x}.nc")(value)
        result = converters.load_dataset(result)
    except DataError as e:
        raise DataError(
            f"Could not load solar irradiance spectrum '{value}'. This is usually "
            "caused by an incorrect path or identifier, or a missing data file. "
            "Do not forget to install the shipped solar irradiance data."
        ) from e

    return result


@define(eq=False, slots=False)
class SolarIrradianceSpectrum(Spectrum):
    """
    Solar irradiance spectrum [``solar_irradiance``].

    This scene element produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this element is controlled by the ``dataset`` field.

    See Also
    --------

    :ref:`Solar irradiance spectrum data guide <sec-data-solar_irradiance>`

    Notes
    ------

    * The spectral range of the data sets shipped can vary and an attempt for
      use outside the supported spectral range will raise a
      :class:`ValueError` upon calling :meth:`kernel_dict`.

    * When the ``datetime`` field is set, the spectrum is automatically scaled
      to account for the seasonal variations of the Earth-Sun distance using the
      ephemeris of :func:`astropy.coordinates.get_sun`.
      The dataset is assumed to be normalized to an Earth-Sun distance of 1 AU.
      This will trigger the import of :mod:`astropy.coordinates` and consume a
      significant amount of memory (150 MiB with astropy v5.1).

    * The ``scale`` field can be used to apply additional arbitrary scaling.
      It is mostly used for debugging purposes. It can also be used to rescale
      user-defined spectra normalized at an Earth-Sun distance different from
      1 AU.

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

    quantity: PhysicalQuantity = attrs.field(
        default=PhysicalQuantity.IRRADIANCE, init=False, repr=False
    )

    dataset: xr.Dataset = documented(
        attrs.field(
            default="coddington_2021-1_nm",
            converter=_dataset_converter,
            validator=attrs.validators.instance_of(xr.Dataset),
            repr=summary_repr,
        ),
        doc="Solar irradiance spectrum dataset. "
        "If an xarray dataset is passed, it is used without changes "
        "(refer to the data guide for format requirements). "
        "If a string is passed, it is interpreted as a solar irradiance "
        "spectrum identifier "
        "(see :ref:`sec-data-solar_irradiance` for the list). "
        "If a path is passed, the converter tries to open the corresponding "
        "file on the hard drive (relative paths are resolved by the file resolver). ",
        type="Dataset",
        init_type="Dataset or str or path-like, optional",
        default='"coddington_2021-1_nm"',
    )

    scale: float = documented(
        attrs.field(default=1.0, converter=float, validator=validators.is_positive),
        doc="Arbitrary scaling factor. This scaling factor is applied in "
        "addition to the datetime-based scaling controlled by the *datetime* "
        "parameter.",
        type="float or datetime",
        init_type="float or datetime or str",
        default="1.0",
    )

    datetime: datetime.datetime | None = documented(
        attrs.field(
            default=None,
            converter=_datetime_converter,
        ),
        type="datetime or None",
        init_type="datetime or str, optional",
        doc="Date for which the spectrum is to be evaluated. An ISO "
        "string can be passed and will be interpreted by "
        ":meth:`dateutil.parser.parse`. This parameter scales the irradiance "
        "spectrum to account for the seasonal variation of the Earth-Sun "
        "distance. This scaling is applied in addition to the arbitrary "
        "scaling controlled by the *scale* parameter.",
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
            # Note: astropy.coordinates consumes a significant amount of memory
            # (150 MiB with astropy v5.1). The import is therefore optional for
            # performance.
            import astropy.coordinates
            import astropy.time
            import astropy.units

            # The irradiance scales as the inverse of d**2, where d is the
            # Earth-Sun distance divided by the AU (reference distance for all
            # Solar irradiance spectra in Eradiate).
            return (
                float(
                    astropy.units.au
                    / astropy.coordinates.get_sun(
                        astropy.time.Time(self.datetime)
                    ).distance
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
            raise ValueError("interpolation of solar irradiance dataset returned nan")

        return irradiance * self.scale * self._scale_earth_sun_distance()

    def eval_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_mono(w=w)

    def integral(self, wmin: pint.Quantity, wmax: pint.Quantity) -> pint.Quantity:
        raise NotImplementedError

    @property
    def template(self) -> dict:
        # Inherit docstring

        return {
            "type": "uniform",
            "value": DictParameter(
                func=lambda ctx: float(self.eval(ctx.si).m_as(uck.get("irradiance")))
            ),
        }

    @property
    def params(self) -> dict:
        # Inherit docstring

        return {
            "value": SceneParameter(
                func=lambda ctx: float(self.eval(ctx.si).m_as(uck.get("irradiance"))),
                flags=KernelSceneParameterFlags.SPECTRAL,
            )
        }
