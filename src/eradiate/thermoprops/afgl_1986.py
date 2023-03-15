"""
Atmospheric thermophysical properties profiles models according to
:cite:`Anderson1986AtmosphericConstituentProfiles`.
"""
from __future__ import annotations

import typing as t

import pint
import xarray as xr

from .util import compute_scaling_factors, interpolate, rescale_concentration
from .. import data
from .._mode import mode

VALID_MODEL_IDS = [
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    "us_standard",
]


def make_profile(
    model_id: str = "us_standard",
    levels: pint.Quantity | None = None,
    concentrations: t.MutableMapping[str, pint.Quantity] | None = None,
) -> xr.Dataset:
    """
    Return atmosphere thermophysical properties based on atmosphere models
    defined in AFGL's 1986 technical report
    :cite:`Anderson1986AtmosphericConstituentProfiles`.

    This function lets you interpolate the atmosphere thermophysical properties
    on a different level altitude mesh than the original.

    You can also rescale the molecules' fractions so that target concentrations
    are met for these molecules.
    Target concentrations can be provided in different units.
    For more information about rescaling process and the supported
    concentration units, refer to the documentation of
    :func:`~eradiate.thermoprops.util.compute_scaling_factors`.

    Parameters
    ----------
    model_id : {"us_standard", "midlatitude_summer", "midlatitude_winter", \
        "subarctic_summer", "subarctic_winter", "tropical"}, \
        optional, default: "us_standard"
        Model identifier.

    levels : quantity or array, optional
        Altitude levels. The array must contain at least two values.
        If not provided, the atmospheric profile is built using the data set's
        altitude levels.

    concentrations : dict, optional
        Molecules concentrations as a {str: quantity} mapping.

    Returns
    -------
    Dataset
        Atmosphere thermophysical properties.

    Notes
    -----
    :cite:`Anderson1986AtmosphericConstituentProfiles` defines six models,
    listed in the table below.

    .. list-table:: AFGL (1986) atmospheric thermophysical properties models
       :widths: 2 4 4
       :header-rows: 1

       * - Model number
         - Model identifier
         - Model name
       * - 1
         - ``tropical``
         - Tropic (15N Annual Average)
       * - 2
         - ``midlatitude_summer``
         - Mid-Latitude Summer (45N July)
       * - 3
         - ``midlatitude_winter``
         - Mid-Latitude Winter (45N Jan)
       * - 4
         - ``subarctic_summer``
         - Sub-Arctic Summer (60N July)
       * - 5
         - ``subarctic_winter``
         - Sub-Arctic Winter (60N Jan)
       * - 6
         - ``us_standard``
         - U.S. Standard (1976)

    .. attention::
       The original altitude mesh specified by
       :cite:`Anderson1986AtmosphericConstituentProfiles` is a piece-wise
       regular altitude mesh with an altitude step of 1 km from 0 to 25 km,
       2.5 km from 25 km to 50 km and 5 km from 50 km to 120 km.
       Since the Eradiate kernel only supports regular altitude mesh, the
       original atmospheric thermophysical properties profiles were
       interpolated on the regular altitude mesh with an altitude step of 1 km
       from 0 to 120 km.

    All six models include the following 10 molecular species:
    H2O, CO2, O3, N2O, CO, CH4, O2, NO, SO2 and NO2.
    """
    if model_id not in VALID_MODEL_IDS:
        raise ValueError(f"model_id should be in {VALID_MODEL_IDS} (got '{model_id}')")
    species = set(concentrations.keys()) if concentrations else set()
    unhandled = species - {"H2O", "CO2", "O3"}

    if unhandled:
        if mode().is_ckd:
            raise NotImplementedError(
                f"molecules '{unhandled}' cannot be rescaled in ckd mode"
            )

    thermoprops = data.load_dataset(f"thermoprops/afgl_1986-{model_id}.nc")

    if levels is not None:
        thermoprops = interpolate(
            ds=thermoprops,
            z_level=levels,
            conserve_columns=True,
        )

    if concentrations is not None:
        factors = compute_scaling_factors(
            ds=thermoprops,
            concentration=concentrations,
        )
        thermoprops = rescale_concentration(ds=thermoprops, factors=factors)

    return thermoprops
