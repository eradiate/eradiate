"""
Data loading, conversion and output components.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pint
import pinttrs
import xarray as xr

from ..data import fresolver
from ..typing import PathLike
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


def _convert_units(value: str):
    if value == "per cent":
        return "percent"
    return value


def load_aerosol_libradtran(
    data: PathLike | xr.Dataset,
    particle_shape: Literal["spherical", "spheroidal"] | None = None,
    tolerance: dict[str, pint.Quantity | float] | None = None,
    wbounds: tuple = (None, None),
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

    particle_shape : {"spherical", "spheroidal"}, optional
        The expected shape of the particle (this will tell the phase matrix
        coefficients it should expect). If unset, the shape is inferred from the
        input dataset.

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

    Returns
    -------
    Dataset

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

    * The current implementation resamples the angular dimension at the
      highest resolution to minimize the loss of information on phase matrix
      coefficients.

    * All conversion is done in memory: very large dataset might result in
      massive converted data. In such case, an easy way to split the conversion
      is to chunk it on the spectral dimension.
    """
    VARS_TO_DIMS = {"wavelen": "nlam", "reff": "nreff", "hum": "nhum"}
    KWARG_TO_DEFAULT_UNITS = {
        "w": ucc.get("wavelength"),
        "hum": ureg.Unit("percent"),
        "reff": ureg.Unit("micrometer"),
    }

    # Load aerosol component dataset
    if not isinstance(data, xr.Dataset):
        data = fresolver.load_dataset(data)

    # Extract required variables
    vars = ["phase", "ext", "ssa", "theta", "wavelen"]
    for var in ["reff", "hum"]:
        if var in data:
            vars.append(var)
    data = data[vars]

    # Select on humidity and effective radius
    if tolerance is None:
        tolerance = {}

    for kwarg in ["hum", "reff"]:
        var = kwarg
        if var not in data:
            continue

        da = data[var]
        units = ureg.Unit(_convert_units(data[var].units))
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
            f"load_aerosol_libradtran() got unexpected keyword arguments {list(kwargs.keys())}, which were not used"
        )

    # Filter wavelengths if requested
    wmin, wmax = wbounds
    units = ureg.Unit(_convert_units(data["wavelen"].units))
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

    wavelength = data["wavelen"].values * ureg(data["wavelen"].units)

    # Phase function
    if particle_shape is None:
        if len(data["nphamat"] == 4):
            particle_shape = "spherical"
        elif len(data["nphamat"] == 6):
            particle_shape = "spheroidal"
        else:
            raise ValueError("Could not detect particle shape type")

    if particle_shape == "spherical":
        ij_to_nphamat = {
            (0, 0): 0,
            (1, 1): 0,
            (0, 1): 1,
            (1, 0): 1,
            (2, 2): 2,
            (3, 3): 2,
            (2, 3): 3,
            (3, 2): 3,
        }
    elif particle_shape == "spheroidal":
        ij_to_nphamat = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 4,
            (2, 2): 2,
            (2, 3): 3,
            (3, 2): 3,
            (3, 3): 5,
        }
    else:
        raise NotImplementedError(f"Unknown particle shape '{particle_shape}'")

    # -- Create angular grid (highest resolution possible)
    mus = np.cos(np.deg2rad(data["theta"].values.ravel()))
    mus = mus[~np.isnan(mus)]
    mus = np.unique(mus)

    # -- Resample all phase matrix components and fill data array with shape
    #    [wavelength, theta, i, j]
    n_wavelength = len(wavelength)
    n_theta = len(mus)
    phase_np = np.zeros((n_wavelength, n_theta, 4, 4))

    for i_wavelength in range(n_wavelength):
        for (i, j), nphamat in ij_to_nphamat.items():
            data_selected = data.isel(nlam=i_wavelength, nphamat=nphamat).dropna(
                "nthetamax"
            )
            x = mus
            xp = data_selected["theta"].values
            xp = np.cos(np.deg2rad(data_selected["theta"].values)).ravel()
            fp = data_selected["phase"].values
            p = np.interp(x, xp, fp)
            phase_np[i_wavelength, :, i, j] = p

    # Populate Eradiate dataset with correct format
    phase_eradiate = xr.Dataset(
        data_vars={
            "sigma_t": (["w"], data["ext"].values, {"units": "1/km"}),
            "albedo": (["w"], data["ssa"].values, {"units": ""}),
            "phase": (["w", "mu", "i", "j"], phase_np),
        },
        coords={
            "w": ("w", wavelength.m_as("nm"), {"units": "nm"}),
            "mu": ("mu", mus),
            "i": ("i", range(4)),
            "j": ("j", range(4)),
        },
    )

    return phase_eradiate


def voxelflux_to_raytran(
    data: xr.Dataset, w: float, export_path: str | None = None
) -> pd.DataFrame:
    """
    Converts the results of a VoxelFluxMeasure to the Raytran format, returned
    in a  ``pandas.DataFrame``. Can optionally save the converted data to a csv
    file by specifying the ``export_path``.

    Parameters
    ----------

    data: xr.Dataset
        Result dataset from a voxelflux measurement.

    w: float
        The wavelength selected for conversion.

    Returns
    -------

    pd.DataFrame
        Result data in the Raytran format.
    """

    # retrieve voxel indexes, and create a meshgrid to be used as multi-index
    x = data.x_index.values[:-1]
    y = data.y_index.values[:-1]
    z = data.z_index.values[:-1]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Xinl = fluxes[0][
    #     1    # positive direction
    #     0,   # x faces
    #     :-1, # x dimension up to last face
    #     :-1, # y dimension on all faces
    #     :-1, # z dimension on all faces
    # ]
    flux = data["flux"].sel(w=w).isel(sza=0, saa=0).values.copy()

    Xinl = flux[1, 0, :-1, :-1, :-1].flatten()
    Xinh = flux[0, 0, 1:, :-1, :-1].flatten()
    Xoutl = flux[0, 0, :-1, :-1, :-1].flatten()
    Xouth = flux[1, 0, 1:, :-1, :-1].flatten()

    Yinl = flux[1, 1, :-1, :-1, :-1].flatten()
    Yinh = flux[0, 1, :-1, 1:, :-1].flatten()
    Youtl = flux[0, 1, :-1, :-1, :-1].flatten()
    Youth = flux[1, 1, :-1, 1:, :-1].flatten()

    Zinl = flux[1, 2, :-1, :-1, :-1].flatten()
    Zinh = flux[0, 2, :-1, :-1, 1:].flatten()
    Zoutl = flux[0, 2, :-1, :-1, :-1].flatten()
    Zouth = flux[1, 2, :-1, :-1, 1:].flatten()

    np_df = np.stack(
        [
            Xinl,
            Xinh,
            Yinl,
            Yinh,
            Zinl,
            Zinh,
            Xoutl,
            Xouth,
            Youtl,
            Youth,
            Zoutl,
            Zouth,
        ]
    )

    columns = [
        "Xinl",
        "Xinh",
        "Yinl",
        "Yinh",
        "Zinl",
        "Zinh",
        "Xoutl",
        "Xouth",
        "Youtl",
        "Youth",
        "Zoutl",
        "Zouth",
    ]

    # Transform to dataframe
    multi_idx = pd.MultiIndex.from_arrays([X, Y, Z], names=["x", "y", "z"])
    df = pd.DataFrame(np_df.T, index=multi_idx, columns=columns)

    # save to csv using Raytran separator?
    if export_path is not None:
        df.to_csv(export_path, sep=" ")

    return df
