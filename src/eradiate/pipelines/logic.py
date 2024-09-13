from __future__ import annotations

import itertools
from collections import OrderedDict

import numpy as np
import pint
import pinttrs
import xarray as xr
from pinttr.util import always_iterable

from .._mode import Mode
from ..exceptions import UnsupportedModeError
from ..kernel import bitmap_to_dataarray
from ..quad import Quad
from ..scenes.illumination import (
    AbstractDirectionalIllumination,
    ConstantIllumination,
    Illumination,
)
from ..scenes.spectra import Spectrum
from ..spectral import BandSRF, SpectralResponseFunction, UniformSRF
from ..spectral.grid import SpectralGrid
from ..units import symbol, to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_context_kernel as uck
from ..units import unit_registry as ureg


def _spectral_dims(mode: Mode | str) -> tuple[tuple[str, dict], ...]:
    """
    Return spectral dimension names and associated metadata depending on the
    passed mode.
    """
    # Note: This is a callable because of dynamic unit symbol evaluation
    w = (
        "w",
        {
            "standard_name": "radiation_wavelength",
            "long_name": "wavelength",
            "units": symbol(ucc.get("wavelength")),
        },
    )
    g = (
        "g",
        {
            "standard_name": "g",
            "long_name": "g",
            "units": symbol(ucc.get("dimensionless")),
        },
    )
    if mode == "mono" or mode.is_mono:
        return (w,)
    elif mode == "ckd" or mode.is_ckd:
        return w, g
    else:
        raise UnsupportedModeError


def aggregate_ckd_quad(
    mode_id: str,
    raw_data: xr.DataArray,
    spectral_grid: SpectralGrid,
    ckd_quads: list[Quad],
    is_variance: bool,
) -> xr.DataArray:
    """
    Compute CKD quadrature.

    In CKD modes, this pipeline step aggregates spectral data and
    evaluates the selected quadrature rule. The following updates to the input
    data are expected:

    * the ``g`` dimension is dropped;
    * a ``bin_wmin`` (resp. ``bin_wmax``) coordinate is created and contains the
      lower (resp. upper) spectral bound of each bin;
    * the dataset is reordered by ascending ``w`` values.

    Parameters
    ----------
    mode_id : str
        Eradiate mode for which this pipeline step is configured. If set to
        "ckd", CKD quadrature computation is performed; otherwise, it is a
        no-op.

    raw_data : DataArray
        A data array holding raw bitmap data indexed against pixel indices.

    spectral_grid : .CKDSpectralGrid
        Spectral set for which the CKD quadrature is computed.

    ckd_quads : list of .Quad
        List of quadrature rules matching the spectral bins held by
        ``spectral_grid``.

    is_variance : bool
        Flag that specifies whether the raw_data is a variance value.

    Returns
    -------
    DataArray
        A data array indexed against spectral coordinates associated with each
        spectral bin.

    Notes
    -----
    * In non-CKD modes, this step is a no-op.
    * During pipeline assembly, this node expands as a single node named
      ``<var>``.
    * The ``spp`` variable is averaged on the ``index`` dimension.
    """
    mode = Mode.new(mode_id)
    result_name = raw_data.name.removesuffix("_raw")

    if not mode.is_ckd:
        result = raw_data.copy()
        result.name = result_name
        return raw_data

    # Get dimensions of current variable
    sizes = OrderedDict((y, len(raw_data.coords[y])) for y in raw_data.sizes)

    for dim in ["w", "g"]:
        if dim not in sizes:
            raise ValueError(
                f"CKD quadrature computation requires dimension {dim}, missing "
                "from input data"
            )

    # Init storage
    del sizes["w"]
    del sizes["g"]

    # -- Collect wavelengths associated with each bin
    wavelength_units = ucc.get("wavelength")
    bin_wmins = spectral_grid.wmins.m_as(wavelength_units)
    bin_wmaxs = spectral_grid.wmaxs.m_as(wavelength_units)

    # -- Proceed with actual storage initialization
    result = xr.full_like(raw_data, np.nan).isel(g=0, drop=True)

    # For each bin and each pixel, compute quadrature and store the result
    for i_bin, (w, quad) in enumerate(zip(spectral_grid.wcenters, ckd_quads)):
        values_at_nodes = raw_data.sel(w=w).values

        # Rationale: Avoid using xarray's indexing in this loop for
        # performance reasons (wrong data indexing method will result in
        # 10x+ speed reduction)
        for indexes in itertools.product(*[list(range(n)) for n in sizes.values()]):
            interval = (0.0, 1.0)

            if is_variance:
                weights = quad.weights.copy()
                if interval is not None:
                    weights *= 0.5 * (interval[1] - interval[0])

                variance = values_at_nodes[(slice(None), *indexes)]
                weighted_sum = float(np.dot(weights**2, variance))
                result.values[(i_bin, *indexes)] = weighted_sum
            else:
                result.values[(i_bin, *indexes)] = quad.integrate(
                    values_at_nodes[(slice(None), *indexes)],
                    interval=interval,
                )

    result.attrs = raw_data.attrs
    result.name = result_name

    # TODO: Provide separate implementation
    # # Average the 'spp' variable over the 'index' dimension
    # with xr.set_options(keep_attrs=True):
    #     result["spp"] = gathered_bitmaps.spp.mean(dim="index")

    # Add spectral coordinates
    result = result.assign_coords(
        {
            "bin_wmin": (
                "w",
                bin_wmins,
                {
                    "standard_name": "bin_wmin",
                    "long_name": "spectral bin lower bound",
                    "units": symbol(wavelength_units),
                },
            ),
            "bin_wmax": (
                "w",
                bin_wmaxs,
                {
                    "standard_name": "bin_wmax",
                    "long_name": "spectral bin upper bound",
                    "units": symbol(wavelength_units),
                },
            ),
        }
    )

    # Reorder by ascending "w"
    result = result.sortby("w")

    return result


def apply_spectral_response(
    spectral_data: xr.DataArray, srf: SpectralResponseFunction
) -> xr.DataArray:
    """
    Apply SRF weighting (a.k.a. convolution) to spectral data and turn it into
    a band aggregate.

    Parameters
    ----------
    spectral_data : DataArray
        Spectral data to process.

    srf : SpectralResponseFunction
        Spectral response function to apply.

    Returns
    -------
    DataArray or None
        A data array where the spectral dimension is removed after applying
        SRF weighting, or ``None`` if the SRF is a :class:`.DeltaSRF`.
    """
    if not {"bin_wmin", "bin_wmax"}.issubset(set(spectral_data.coords.keys())):
        raise ValueError(
            "input data is missing 'bin_wmin' and/or 'bin_wmax' coordinates"
        )

    # Evaluate integral of spectral response function within selected interval
    wmin = to_quantity(spectral_data.coords["bin_wmin"]).min()
    wmax = to_quantity(spectral_data.coords["bin_wmax"]).max()
    srf_int = srf.integrate(wmin, wmax)

    if isinstance(srf, BandSRF):
        srf_w = srf.wavelengths
    elif isinstance(srf, UniformSRF):
        srf_w = np.array([wmin.m_as(ureg.nm), wmax.m_as(ureg.nm)]) * ureg.nm
    else:
        raise TypeError(f"unhandled SRF type '{srf.__class__.__name__}'")

    # Evaluate integral of product of variable and SRF within selected interval
    data_w = to_quantity(spectral_data.coords["w"])

    # Spectral grid is the finest between data and SRF grids
    w_units = data_w.units
    w_m = np.array(sorted(set(data_w.m_as(w_units)) | set(srf_w.m_as(w_units))))

    # If data var has length 1 on spectral dimension, directly select
    # the value instead of using interpolation (it's a known scipy issue)
    if len(spectral_data.coords["w"]) == 1:
        # Note: The tricky thing is to recreate and extend the 'w'
        # dimension with the same axis index as in the original data
        spectral_values = spectral_data.isel(w=0, drop=True).expand_dims(
            w=w_m, axis=spectral_data.get_axis_num("w")
        )

    # Otherwise, use nearest neighbour interpolation (we assume that the
    # spectral data is constant over each spectral bin)
    else:
        spectral_values = spectral_data.interp(
            w=w_m, method="nearest", kwargs={"fill_value": "extrapolate"}
        )

    srf_values = (
        srf.eval(w_m * w_units)
        .reshape([-1 if dim == "w" else 1 for dim in spectral_values.dims])
        .magnitude
    )
    assert isinstance(srf_values, np.ndarray)  # Check for leftover bugs
    var_srf_int = (spectral_values * srf_values).integrate("w")

    # Initialize storage (we want to keep all coordinate variables)
    result = xr.full_like(spectral_data, np.nan).isel(w=0, drop=True)
    result.values = var_srf_int.values / srf_int.m_as(w_units)

    # Apply SRF to variable and store result
    attrs = spectral_values.attrs.copy()
    if "standard_name" in attrs:
        attrs["standard_name"] += "_srf"
    if "long_name" in attrs:
        attrs["long_name"] += " (SRF-weighted)"

    try:
        name = spectral_data.name + "_srf"
    except TypeError as e:
        raise TypeError("expected a DataArray with a name") from e

    result.attrs = attrs
    result.name = name

    return result


def compute_albedo(
    radiosity_data: xr.DataArray, irradiance_data: xr.DataArray
) -> xr.DataArray:
    """
    Compute the bi-hemispherical reflectance (a.k.a surface albedo) from
    radiosity and irradiance records.

    Parameters
    ----------
    radiosity_data : DataArray
        Radiosity record.

    irradiance_data : DataArray
        Irradiance record.

    Returns
    -------
    dict
        A dictionary with the following content:

        ``albedo`` : :class:`xarray.DataArray`
            An albedo data array.
    """

    # We assume that all quantities are stored in kernel units
    result = radiosity_data / irradiance_data
    result.name = "albedo"
    result.attrs = {
        "standard_name": "albedo",
        "long_name": "surface albedo",
        "units": "",
    }

    return result


def compute_bidirectional_reflectance(
    radiance_data: xr.DataArray, irradiance_data: xr.DataArray, calculate_stokes: bool
) -> dict[str, xr.DataArray]:
    """
    Compute the spectral bidirectional reflectance distribution function
    (BRDF) and bidirectional reflectance factor (BRF) from radiance and
    irradiance records.

    Parameters
    ----------
    radiance_data : DataArray
        Radiance record.

    irradiance_data : DataArray
        Irradiance record.

    calculate_stokes : bool
        Specifies whether the radiance is a Stokes vector

    Returns
    -------
    dict
        A dictionary with the following content:

        ``brdf`` : :class:`xarray.DataArray`
            A BRDF data array.
        ``brf`` : :class:`xarray.DataArray`
            A BRF data array.

    Notes
    -----
    This function is expanded to ``brdf`` and ``brf`` nodes during pipeline
    assembly.
    """
    # We assume that all quantities are stored in kernel units

    radiance = radiance_data
    if calculate_stokes:
        radiance = radiance_data.sel(stokes="I")

    brdf_data = radiance / irradiance_data
    brdf_data.attrs = {
        "standard_name": "brdf",
        "long_name": "bi-directional reflection distribution function",
        "units": symbol("1/sr"),
    }
    brdf_data.name = "brdf"

    brf_data = brdf_data * np.pi
    brf_data.attrs = {
        "standard_name": "brf",
        "long_name": "bi-directional reflectance factor",
        "units": symbol("dimensionless"),
    }
    brf_data.name = "brf"

    return {"brdf": brdf_data, "brf": brf_data}


def extract_irradiance(
    mode_id: str, illumination: Illumination, spectral_grid: SpectralGrid
) -> dict:
    """
    Derive an irradiance dataset from the irradiance spectrum of an illuminant,
    as well as solar angles.

    Parameters
    ----------
    mode_id : str
        Eradiate mode for which this pipeline step is to be configured.

    illumination : Illumination
        The illuminant whose irradiance is to be evaluated.

    spectral_grid : .SpectralGrid
        Spectral grid driving the simulation.

    Returns
    -------
    dict
        A dictionary with the following content:

        ``irradiance`` : :class:`xarray.DataArray`
          A spectral irradiance data array compatible with processed spectral
          radiance records and suitable for deriving BRDF data.
        ``solar_angles`` : :class:`xarray.Dataset` or ``None``
          If relevant, a dataset holding solar angles.

    Notes
    -----
    This function is expanded into ``irradiance`` and ``solar_angles`` nodes
    during pipeline assembly.
    """
    mode = Mode.new(mode_id)
    k_irradiance_units = uck.get("irradiance")
    wavelengths = spectral_grid.wavelengths
    wavelength_metadata = _spectral_dims("mono")[0][1]
    wavelength_units = wavelength_metadata["units"]

    def eval_illumination_spectrum(
        field_name: str, k_units: pint.Unit
    ) -> pint.Quantity:
        # Local helper function to evaluate the illumination spectrum

        spectrum: Spectrum = getattr(illumination, field_name)

        if mode.is_mono:
            return spectrum.eval_mono(wavelengths).m_as(k_units)

        elif mode.is_ckd:
            # Note: We pass here None to the g parameter because irradiance
            # spectrum data never have a g coordinate: the g parameter is simply
            # ignored.
            result = spectrum.eval_ckd(w=wavelengths, g=None).m_as(k_units)

            # Reorder data by ascending wavelengths
            indices = wavelengths.argsort()

            return result[indices]

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    # Now, proceed with actual spectrum evaluation
    solar_angles = None

    # Define spectral coordinates
    spectral_coords = {
        "w": ("w", wavelengths.m_as(wavelength_units), wavelength_metadata)
    }

    if mode.is_ckd:
        bin_wmins = spectral_grid.wmins.m_as(wavelength_units)
        bin_wmaxs = spectral_grid.wmaxs.m_as(wavelength_units)
        spectral_coords.update(
            {
                "bin_wmin": (
                    "w",
                    bin_wmins,
                    {
                        "standard_name": "bin_wmin",
                        "long_name": "spectral bin lower bound",
                        "units": symbol(wavelength_units),
                    },
                ),
                "bin_wmax": (
                    "w",
                    bin_wmaxs,
                    {
                        "standard_name": "bin_wmax",
                        "long_name": "spectral bin upper bound",
                        "units": symbol(wavelength_units),
                    },
                ),
            }
        )

    if isinstance(illumination, AbstractDirectionalIllumination):
        # Collect illumination angular data
        saa = illumination.azimuth.m_as(ureg.deg)
        sza = illumination.zenith.m_as(ureg.deg)
        cos_sza = np.cos(np.deg2rad(sza))

        # Package solar angles into a dataset
        solar_angles = xr.Dataset(
            {
                "sza": (
                    "sza",
                    [sza],
                    {
                        "standard_name": "solar_zenith_angle",
                        "long_name": "solar zenith angle",
                        "units": symbol("deg"),
                    },
                ),
                "saa": (
                    "saa",
                    [saa],
                    {
                        "standard_name": "solar_azimuth_angle",
                        "long_name": "solar azimuth angle",
                        "units": symbol("deg"),
                    },
                ),
            }
        )

        # Collect illumination spectral data
        irradiances = eval_illumination_spectrum("irradiance", k_irradiance_units)

        # Create irradiance variable
        irradiance = xr.DataArray(
            data=(irradiances * cos_sza).reshape((1, 1, len(irradiances))),
            dims=("sza", "saa", "w"),
            coords={
                "sza": solar_angles["sza"],
                "saa": solar_angles["saa"],
                **spectral_coords,
            },
            name="irradiance",
        )

    elif isinstance(illumination, ConstantIllumination):
        # Collect illumination spectral data
        k_radiance_units = uck.get("radiance")
        radiances = eval_illumination_spectrum("radiance", k_radiance_units)

        # Create irradiance variable
        irradiance = xr.DataArray(
            data=np.pi * radiances.reshape((len(radiances),)),
            dims=("w",),
            coords=spectral_coords,
            name="irradiance",
        )

    else:
        raise TypeError(
            "field 'illumination' must be one of "
            "(DirectionalIllumination, ConstantIllumination), got a "
            f"{illumination.__class__.__name__}"
        )

    irradiance.attrs = {
        "standard_name": "horizontal_solar_irradiance_per_unit_wavelength",
        "long_name": "horizontal spectral irradiance",
        "units": symbol(k_irradiance_units),
    }

    return {"irradiance": irradiance, "solar_angles": solar_angles}


def gather_bitmaps(
    mode_id: str,
    var_name: str,
    var_metadata: dict,
    gather_variance: bool,
    calculate_stokes: bool,
    bitmaps: dict,
    viewing_angles: xr.Dataset,
    solar_angles: xr.Dataset,
) -> dict:
    """
    Gather a collection of Mitsuba bitmaps into xarray data arrays.

    Parameters
    ----------
    mode_id : str
        Eradiate mode from which this pipeline step is configured.

    var_name : str
        Name of the processed physical variable.

    var_metadata : dict
        A metadata dictionary to be attached to the data array holding the
        processed physical variable.

    calculate_stokes : bool
        Flag that specifies whether the variable is calculated as a Stokes
        vector or not.

    gather_variance : bool
        Flag that specifies whether the variance bitmaps should be gathered.

    bitmaps : dict
        A dictionary mapping spectral loop indexes to the corresponding bitmap.
        Dictionary structure is as follows:

        .. code::

           {
               <loop_index_0>: {
                   {"bitmap": <bitmap_0>},
                   {"spp": <sample_count_0>},
               },
               <loop_index_1>: {
                   {"bitmap": <bitmap_1>},
                   {"spp": <sample_count_1>},
               },
               ...
           }

    viewing_angles : Dataset, optional
        A dataset holding the viewing angles associated with each pixel in the
        processed bitmaps.

    solar_angles : Dataset, optional
        A dataset holding the solar angles associated with the processed
        observation data.

    Returns
    -------
    data_vars : dict[str, DataArray]
        A dictionary mapping data variable names to a corresponding data array.
        These can easily be aggregated into an xarray dataset, or scattered
        around other nodes of the post-processing pipeline.
    """
    mode = Mode.new(mode_id)

    # Set up spectral dimensions
    spectral_dims = []
    spectral_dim_metadata = {}

    for y in _spectral_dims(mode):
        if isinstance(y, str):
            spectral_dims.append(y)
            spectral_dim_metadata[y] = {}
        else:
            spectral_dims.append(y[0])
            spectral_dim_metadata[y[0]] = y[1]

    # Loop on spectral indexes and collect all bitmap contents in data arrays
    sensor_data = {
        "spp": [],
        "weights_raw": [],
        f"{var_name}_raw": [],
        f"{var_name}_m2_raw": [],
    }

    for spectral_index_hashable, result_dict in bitmaps.items():
        spectral_index = spectral_index_hashable

        # Set spectral coordinates
        all_coords = {
            spectral_dim: [spectral_coord]
            for spectral_dim, spectral_coord in zip(
                spectral_dims, always_iterable(spectral_index)
            )
        }

        spp = result_dict["spp"]

        # Package spp in a data array
        all_dims = list(all_coords.keys())
        spp_shape = [1 for _ in all_dims]
        sensor_data["spp"].append(
            xr.DataArray(np.reshape(spp, spp_shape), coords=all_coords)
        )

        if not calculate_stokes:
            name = "bitmap"
            da = bitmap_to_dataarray(result_dict[name])

            # Add spectral and sensor dimensions to img array
            sensor_data[f"{var_name}_raw"].append(da.expand_dims(dim=all_coords))

            if gather_variance:
                name = "m2"
                da_m2 = bitmap_to_dataarray(result_dict[name])
                sensor_data[f"{var_name}_m2_raw"].append(
                    da_m2.expand_dims(dim=all_coords)
                )

        else:
            components = ["I", "Q", "U", "V"]
            stokes = []
            for s in components:
                stokes.append(bitmap_to_dataarray(result_dict[s]))
            da = xr.concat(stokes, "stokes")
            da = da.assign_coords({"stokes": components})

            # Add spectral and sensor dimensions to img array
            sensor_data[f"{var_name}_raw"].append(da.expand_dims(dim=all_coords))

            if gather_variance:
                m2_components = ["m2_" + s for s in components]
                stokes_m2 = []
                for s in m2_components:
                    stokes_m2.append(bitmap_to_dataarray(result_dict[s]))
                da_m2 = xr.concat(stokes_m2, "stokes")
                da_m2 = da_m2.assign_coords({"stokes": components})

                sensor_data[f"{var_name}_m2_raw"].append(
                    da_m2.expand_dims(dim=all_coords)
                )

    # Combine all the data
    result = {k: xr.combine_by_coords(v) if v else None for k, v in sensor_data.items()}

    keys = [f"{var_name}_raw"]
    if gather_variance:
        keys.append(f"{var_name}_m2_raw")

    for key in keys:
        # Add viewing angle coordinates to main data array
        if viewing_angles is not None:
            result[key] = result[key].assign_coords(viewing_angles)

        # Add solar angle coordinate to main data array
        if solar_angles is not None:
            result[key] = result[key].expand_dims(
                {k: solar_angles[k].values for k in ["sza", "saa"]}, axis=(-1, -2)
            )

    # Drop "channel" dimension when using a monochromatic Mitsuba variant
    if mode.check(mi_color_mode="mono"):
        for k, v in result.items():
            if v is not None and "channel" in v.dims:
                result[k] = v.squeeze("channel", drop=True)

    # Apply spectral metadata
    for da in result.values():
        if da is not None:
            for spectral_dim in spectral_dims:
                da[spectral_dim].attrs = spectral_dim_metadata[spectral_dim]

    # Update name of all variables
    for name, da in result.items():
        if da is not None:
            da.name = name

    # Apply metadata to data variables
    result[f"{var_name}_raw"].attrs.update(var_metadata)

    return result


def radiosity(sector_radiosity: xr.DataArray) -> xr.DataArray:
    """
    Aggregate sector radiosity into a full-hemisphere radiosity dataset.

    Parameters
    ----------
    sector_radiosity : DataArray
        Radiosity values for the hemisphere sector corresponding to each film
        pixel. This quantity is expressed in spectral flux units
        (typically W/m²/nm) and, when summed over the entire film, aggregates
        into a spectral radiosity (W/m²/nm).

    Returns
    -------
    DataArray
        A global radiosity record, with no film pixel / viewing angle indexing
        left.
    """
    result = sector_radiosity.sum(dim=("x_index", "y_index"))
    result.attrs = {
        "standard_name": "toa_outgoing_flux_density_per_unit_wavelength",
        "long_name": "top-of-atmosphere outgoing spectral flux density",
        "units": symbol(uck.get("irradiance")),
    }

    return result


def spectral_response(srf: SpectralResponseFunction) -> xr.DataArray:
    """
    Evaluate a spectral response function as a data array.

    Parameters
    ----------
    srf : Spectrum
        The spectral function to be evaluated.

    Returns
    -------
    DataArray
        A data array mapping SRF values against the wavelength. To avoid
        confusion with the spectral coordinate, the wavelength dimension is here
        named ``srf_w``.
    """
    # Evaluate SRF
    w_units = ucc.get("wavelength")

    if isinstance(srf, BandSRF):
        srf_w = srf.wavelengths
        srf_values = pinttrs.util.ensure_units(srf.values, ureg.dimensionless)

    else:
        raise TypeError(f"unsupported SRF type '{srf.__class__.__name__}'")

    result = xr.DataArray(
        data=srf_values.m,
        dims=("srf_w",),
        coords={
            "srf_w": (
                "srf_w",
                srf_w.m_as(w_units),
                {
                    "standard_name": "radiation_wavelength",
                    "long_name": "wavelength",
                    "units": symbol(w_units),
                },
            )
        },
        attrs={
            "standard_name": "spectral_response_function",
            "long_name": "spectral response function",
            "units": "",
        },
    )

    return result


def viewing_angles(angles: np.ndarray) -> xr.Dataset:
    """
    Collect viewing angles associated with each film pixel from the measure, if
    relevant.

    Parameters
    ----------
    angles : ndarray
        An array with shape (m, n, 2) containing the processed measure's viewing
        angles in degree. Typically obtained by calling
        :meth:`.DistantMeasure.viewing_angles`.

    Returns
    -------
    Dataset
        An xarray dataset holding viewing angle values indexed by film
        coordinates.
    """
    theta = angles[:, :, 0]
    phi = angles[:, :, 1]

    return xr.Dataset(
        {
            "vza": xr.DataArray(
                theta,
                coords={
                    "x_index": [i for i in range(theta.shape[0])],
                    "y_index": [i for i in range(theta.shape[1])],
                },
                dims=("x_index", "y_index"),
                attrs={
                    "standard_name": "viewing_zenith_angle",
                    "long_name": "viewing zenith angle",
                    "units": symbol("deg"),
                },
            ),
            "vaa": xr.DataArray(
                phi,
                coords={
                    "x_index": [i for i in range(phi.shape[0])],
                    "y_index": [i for i in range(phi.shape[1])],
                },
                dims=("x_index", "y_index"),
                attrs={
                    "standard_name": "viewing_azimuth_angle",
                    "long_name": "viewing azimuth angle",
                    "units": symbol("deg"),
                },
            ),
        }
    )


def moment2_to_variance(
    expectation: xr.DataArray, m2: xr.DataArray, spp: xr.DataArray
) -> xr.DataArray:
    """
    Calculate the variance (central 2nd moment) from the raw 2nd moment and expected value.

    Parameters
    ----------
    expectation : DataArray
        The data expected value. For Monte Carlo integration, this is the output of
        the estimator.

    m2 : DataArray
        The data's raw 2nd moment.

    spp : DataArray
        The number of samples per pixels used during the Monte Carlo integration.

    Returns
    -------
    DataArray
        The variance (central 2nd moment) of the data. The DataArray has the same shape
        as the expectation or m2.

    Notes
    -----
    This function calculates the Monte-Carlo variance, which aggregates the variance
    of each sample. It thus needs to be divided by the spp.
    """
    variance = (m2 - expectation * expectation) / spp
    variance.name = m2.name.replace("m2", "var")
    return variance


def degree_of_linear_polarization(stokes_vec: xr.DataArray):
    """
    Calculate the degree of linear polarization from a Stokes vector.

    Parameters
    ----------
    stokes_vec : DataArray
        The data's Stokes vector ([I,Q,U,V]).

    Returns
    -------
    DataArray
        Degree of linear polarization (:math:`\\sqrt{Q^2+U^2}/I`).
    """
    return np.sqrt(
        stokes_vec.sel(stokes="Q") ** 2 + stokes_vec.sel(stokes="U") ** 2
    ) / stokes_vec.sel(stokes="I")
