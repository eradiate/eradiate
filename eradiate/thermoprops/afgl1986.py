"""
Atmospheric thermophysical properties profiles models according to
:cite:`Anderson1986AtmosphericConstituentProfiles`.
"""
from typing import MutableMapping, Optional

import pint
import xarray as xr

from .util import compute_scaling_factors, interpolate, rescale_concentration
from ..data import open


def make_profile(
    model_id: str = "us_standard",
    levels: Optional[pint.Quantity] = None,
    concentrations: Optional[MutableMapping[str, pint.Quantity]] = None,
) -> xr.Dataset:
    """Makes the atmospheric profiles from the AFGL's 1986 technical report
    :cite:`Anderson1986AtmosphericConstituentProfiles`.

    :cite:`Anderson1986AtmosphericConstituentProfiles` defines six models,
    listed in the table below.

    .. list-table:: AFGL (1986) atmospheric thermophysical properties profiles models
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

    Although the altitude meshes of the interpolated
    :cite:`Anderson1986AtmosphericConstituentProfiles` profiles is fixed,
    this function lets you define a custom altitude mesh (regular or irregular).

    All six models include the following six absorbing molecular species:
    H2O, CO2, O3, N2O, CO, CH4 and O2.
    The concentrations of these species in the atmosphere is fixed by
    :cite:`Anderson1986AtmosphericConstituentProfiles`.
    However, this function allows you to rescale the concentrations of each
    individual molecular species to custom concentration values.
    Custom concentrations can be provided in different units.
    For more information about rescaling process and the supported
    concentration units, refer to the documentation of
    :func:`~eradiate.thermoprops.util.compute_scaling_factors`.

    Parameter ``model_id`` (str):
        Choose from ``"midlatitude_summer"``, ``"midlatitude_winter"``,
        ``"subarctic_summer"``, ``"subarctic_winter"``, ``"tropical"`` and
        ``"us_standard"``.

        Default: ``"us_standard"``

    Parameter ``levels`` (:class:`~pint.Quantity`):
        Altitude levels.

    Parameter ``concentrations`` (Dict[str, :class:`~pint.Quantity`]):
        Molecules concentrations.

    Returns → :class:`~xarray.Dataset`:
        Atmospheric profile.
    """
    thermoprops = open(category="thermoprops_profiles", id="afgl1986-" + model_id)
    if levels is not None:
        thermoprops = interpolate(ds=thermoprops, z_level=levels, conserve_columns=True)

    if concentrations is not None:
        factors = compute_scaling_factors(ds=thermoprops, concentration=concentrations)
        thermoprops = rescale_concentration(ds=thermoprops, factors=factors)

    return thermoprops
