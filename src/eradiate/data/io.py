"""
Data loading, conversion and output components.
"""

from __future__ import annotations

import warnings

import numpy as np
import pint
import pinttrs
import xarray as xr

from ._file_resolver import fresolver
from .convert import libradtran_to_aer_core_v2
from ..typing import PathLike
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


def _convert_units(value: str):
    if value == "per cent":
        return "percent"
    return value


def _get_units(ds, var, fallback_units=None):
    if "units" in ds[var].attrs:
        return ureg.Unit(_convert_units(ds[var].attrs["units"]))
    elif fallback_units is not None and var in fallback_units:
        return ureg.Unit(_convert_units(fallback_units[var]))
    else:
        raise ValueError(
            "load_aerosol_libradtran(): The input dataset specifies no units "
            f"for variable '{var}'; this can be addressed by passing them through "
            "the 'fallback_units' parameter."
        )


def load_aerosol_libradtran(
    data: PathLike | xr.Dataset,
    tolerance: dict[str, pint.Quantity | float] | None = None,
    wbounds: tuple = (None, None),
    fallback_units: dict[str, str] | None = None,
    **kwargs,
) -> xr.Dataset:
    """
    Convert a libRadtran NetCDF aerosol file to the Eradiate aerosol data format.
    Both effective radius and humidity-indexed data are supported.

    Parameters
    ----------
    data : Dataset or path-like
        A libRadtran NetCDF aerosol dataset. If a path is passed, it will be
        resolved by the file resolver and tentatively loaded into memory.

    reff, hum : float or quantity
        For datasets with a humidity or effective radius dimension, the
        coordinate point to select. By default, the nearest data point is
        selected; tolerance can be adjusted using the ``tolerance`` parameter.
        Quantities are accepted (see Notes for expected dimensions and
        defaults). These parameters are optional if the dataset has only one
        point on these dimensions.

    wbounds : tuple, optional
        Bounds to restrict the spectral domain where conversion is performed.
        This parameter accepts a tuple with the minimum and maximum values
        (use ``None`` to leave a bound open). Examples:
        ``(300, 3000)``, ``(None, 3000)``, ``(0.3, 3) * ureg.micron``,
        ``(None, 3 * ureg.micron)``.

    tolerance : dict
        A mapping that allows to specify a tolerance for nearest neighbour
        lookup for relevant parameters. Units are applied with the same rules as
        for the ``reff`` and ``hum`` parameters.

    fallback_units : dict, optional
        A mapping that specifies units to apply to variables that are missing them.

    Returns
    -------
    Dataset

    See Also
    --------
    :func:`.libradtran_to_aer_core_v2`

    Notes
    -----
    .. list-table::
        :widths: 1 2 2
        :header-rows: 1

        * - Parameter
          - Dimension
          - Default units
        * - ``reff``
          - Length
          - μm
        * - ``hum``
          - Dimensionless (fraction)
          - percent
        * - ``wbounds``
          - Wavelength
          - ``ucc.get("wavelength")`` (usually, nm)

    * All conversion is done in memory: very large datasets might result in
      massive converted data. In such cases, an easy way to split the conversion
      is to chunk it on the spectral dimension.
    """
    # Map variables to their associated dimension names
    VARS_TO_DIMS = {"wavelen": "nlam", "reff": "nreff", "hum": "nhum"}

    # Map kwargs to associated default units in case non-quantities are passed
    KWARG_TO_DEFAULT_UNITS = {
        "w": ucc.get("wavelength"),
        "hum": ureg.Unit("percent"),
        "reff": ureg.Unit("micrometer"),
    }

    # Load aerosol component dataset
    if not isinstance(data, xr.Dataset):
        data = fresolver.load_dataset(data)

    # Extract required variables
    vars = ["phase", "ext", "ssa", "theta", "wavelen", "ntheta", "pmom", "nmom"]
    vars.extend([x for x in ["reff", "hum"] if x in data])
    data = data[vars]

    # Select on humidity and effective radius
    if tolerance is None:
        tolerance = {}

    for kwarg in ["hum", "reff"]:
        var = kwarg
        dim = VARS_TO_DIMS[var]
        if dim not in data.dims:  # skip if dimension is missing
            continue

        units = _get_units(data, var, fallback_units)
        da = data[var]
        default_units = KWARG_TO_DEFAULT_UNITS[kwarg]

        if len(da) > 1:
            if kwarg not in kwargs:
                raise TypeError(
                    f"load_aerosol_libradtran() is missing keyword argument '{kwarg}' "
                    f"(allowed: {da.values})"
                )
            else:
                kwarg_value = pinttrs.converters.ensure_units(
                    np.atleast_1d(kwargs.pop(kwarg)), default_units=default_units
                ).to(units)
        else:
            if kwarg not in kwargs:
                kwarg_value = da.values * units
            else:
                kwarg_value = pinttrs.converters.ensure_units(
                    np.atleast_1d(kwargs.pop(kwarg)), default_units=default_units
                ).to(units)

        sel_kwargs = {var: kwarg_value.m, "method": "nearest"}
        if kwarg in tolerance:
            sel_kwargs["tolerance"] = tolerance[kwarg]

        data = data.swap_dims({VARS_TO_DIMS[var]: var})
        data = data.sel(**sel_kwargs)
        data = data.squeeze(var, drop=True)

    if kwargs:
        warnings.warn(
            "load_aerosol_libradtran() got unexpected keyword arguments "
            f"{list(kwargs.keys())}, which were not used",
            stacklevel=2,
        )

    # Filter wavelengths if requested
    wmin, wmax = wbounds

    units = _get_units(data, "wavelen", fallback_units)
    default_units = KWARG_TO_DEFAULT_UNITS["w"]

    if wmin is not None:
        wmin = pinttrs.converters.ensure_units(wmin, default_units=default_units).m_as(
            units
        )
        data = data.where(data["wavelen"] >= wmin).dropna("nlam", how="all")
    if wmax is not None:
        wmax = pinttrs.converters.ensure_units(wmax, default_units=default_units).m_as(
            units
        )
        data = data.where(data["wavelen"] <= wmax).dropna("nlam", how="all")

    return libradtran_to_aer_core_v2(data, fallback_units=fallback_units)
