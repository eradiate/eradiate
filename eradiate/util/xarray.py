"""xarray-related components."""

import xarray as xr


def eo_dataarray(data, sza, saa, vza, vaa, wavelength):
    """Creates a :class:`~xarray.DataArray` object with the typical coordinates
    and their attributes for EO applications.

    Parameter ``data`` (array):
        Data to fill the created :class:`~xarray.DataArray`.

    Parameter ``sza`` (array):
        Iterable container of Sun zenith angles.

    Parameter ``saa`` (array):
        Iterable container of Sun azimuth angles.

    Parameter ``vza`` (array):
        Iterable container of viewing zenith angles.

    Parameter ``vaa`` (array):
        Iterable container of viewing azimuth angles.

    Parameter ``wavelength`` (float):
        Iterable container of wavelength values.

    Returns → :class:`xarray.DataArray`:
        A data array with coordinates and attributes set to match typical
        EO applications.
    """

    da = xr.DataArray(data, dims=["sza", "saa", "vza", "vaa", "wavelength"],
                      coords={"sza": sza, "saa": saa,
                              "vza": vza, "vaa": vaa,
                              "wavelength": wavelength})

    da.attrs["angle_convention"] = "eo_scene"

    da.sza.attrs["unit"] = "deg"
    da.sza.attrs["long_name"] = "Sun zenith angle"

    da.saa.attrs["unit"] = "deg"
    da.saa.attrs["long_name"] = "Sun azimuth angle"

    da.vza.attrs["unit"] = "deg"
    da.vza.attrs["long_name"] = "Viewing zenith angle"

    da.vaa.attrs["unit"] = "deg"
    da.vaa.attrs["long_name"] = "Viewing azimuth angle"

    da.wavelength.attrs["long_name"] = "Wavelength"
    da.wavelength.attrs["unit"] = "nm"

    return da
