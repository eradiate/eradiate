"""xarray-related components."""

import xarray as xr


def eo_dataarray(data, sza, saa, vza, vaa, wavelength, angular_domain):
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

    Parameter ``angular_domain`` (string):
        Descriptor for different types of measures, e.g. `hsphere` or
        `pplane`.

    Returns → :class:`xarray.DataArray`:
        A data array with coordinates and attributes set to match typical
        EO applications.
    """

    da = xr.DataArray(data, dims=["sza", "saa", "vza", "vaa", "wavelength"],
                      coords={"sza": sza, "saa": saa,
                              "vza": vza, "vaa": vaa,
                              "wavelength": wavelength})

    da.attrs["angular_type"] = "observation"
    da.attrs["angular_domain"] = angular_domain

    da.sza.attrs["units"] = "deg"
    da.sza.attrs["long_name"] = "Sun zenith angle"

    da.saa.attrs["units"] = "deg"
    da.saa.attrs["long_name"] = "Sun azimuth angle"

    da.vza.attrs["units"] = "deg"
    da.vza.attrs["long_name"] = "Viewing zenith angle"

    da.vaa.attrs["units"] = "deg"
    da.vaa.attrs["long_name"] = "Viewing azimuth angle"

    da.wavelength.attrs["long_name"] = "Wavelength"
    da.wavelength.attrs["units"] = "nm"

    return da


# TODO: transfer to xarray decorator
def check_var_metadata(data_set, name, units, standard_name):
    r"""Checks that a data variable/coordinate in a data set has the valid units
    and standard name.

    Raises a `ValueError` if the metadata is incorrect.

    Parameter ``data_set`` (:class:`~xr.Dataset`):
        Input data set.

    Parameter ``name`` (str):
        Data variable/coordinate name.

    Parameter ``units`` (str):
        Expected units.

    Parameter ``standard_name`` (str):
        Expected standard name.
    """

    try:
        da = data_set[name]
        metadata = da.attrs

        try:
            u = metadata["units"]
            if u != units:
                raise ValueError(f"{name} has the wrong units ({u} instead of"
                                 f"{units}).")
        except KeyError:
            raise ValueError(f"{name} does not have units.")

        try:
            s = metadata["standard_name"]
            if s != standard_name:
                raise ValueError(f"{name} has the wrong standard name ({s} "
                                 f"instead of {standard_name})")
        except KeyError:
            raise ValueError(f"{name} does not have a standard name.")

    except AttributeError:
        raise ValueError(f"{name} is not a data variable/coordinate of this "
                         f"data set.")


# TODO: transfer to xarray decorator
def check_metadata(data_set):
    r"""Checks that a data set has valid metadata.

    Raises a ValueError if the data set's metadata are not valid.

    Parameter ``data_set`` (:class:`xr.Dataset`):
        Data set to check.
    """

    for x in ["convention", "title", "history", "source", "references"]:
        try:
            assert x in data_set.attrs
            if data_set.attrs[x] == "":
                raise ValueError(f"The metadata field {x} is empty.")
        except AssertionError:
            raise ValueError(f"The metadata field {x} is missing.")

    # additional attributes are allowed
