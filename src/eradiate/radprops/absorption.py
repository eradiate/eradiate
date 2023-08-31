"""
Functions to compute monochromatic absorption.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pint
import portion as P
import xarray as xr

import eradiate

from ..exceptions import (
    InterpolationError,
    MissingCoordinateError,
    OutOfBoundsCoordinateError,
    ScalarCoordinateError,
    UnsupportedModeError,
)
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
#                       Interpolation errors handling
# ------------------------------------------------------------------------------

DEFAULT_X_ERROR_HANDLER_CONFIG = {
    "missing": "ignore",
    "scalar": "ignore",
    "bounds": "raise",
}

DEFAULT_P_ERROR_HANDLER_CONFIG = {
    "missing": "raise",
    "scalar": "raise",
    "bounds": "ignore",  # atmospheric presure easily goes very low, but at
    # these altitude, the air is very thin and so is the
    # absorption coefficient
}

DEFAULT_T_ERROR_HANDLER_CONFIG = {
    "missing": "raise",
    "scalar": "raise",
    "bounds": "ignore",  # atmospheric temperature climbs fast above 80-100 km,
    # but at these altitude, the air is very thin and so is
    # the absorption coefficient
}

DEFAULT_HANDLER_CONFIG = {
    "x": DEFAULT_X_ERROR_HANDLER_CONFIG,
    "p": DEFAULT_P_ERROR_HANDLER_CONFIG,
    "t": DEFAULT_T_ERROR_HANDLER_CONFIG,
}

ERROR_TYPE_TO_STR = {
    MissingCoordinateError: "missing",
    ScalarCoordinateError: "scalar",
    OutOfBoundsCoordinateError: "bounds",
}


def handle_error(error: InterpolationError, config):
    action = config[ERROR_TYPE_TO_STR[type(error)]]

    if action == "raise":
        raise error
    elif action == "warn":
        warnings.warn(str(error), UserWarning)
    elif action == "ignore":
        pass
    else:
        raise NotImplementedError(f"Action {action} not implemented.")


# ------------------------------------------------------------------------------
#                       Interpolation functions
# ------------------------------------------------------------------------------


def xcoords(k: xr.DataArray) -> list[str]:
    return [c for c in k.coords if c.startswith("x_")]


def xrange(k: xr.DataArray) -> dict[str, pint.Quantity]:
    _xrange = {}
    for c in xcoords(k):
        if k[c].size > 1:
            x = to_quantity(k[c])
            _xrange[c] = np.stack([x.min(), x.max()])
    return _xrange


def xcoords_scalar(k: xr.DataArray) -> list[str]:
    return [c for c in xcoords(k) if k[c].size == 1]


def prange(k: xr.DataArray) -> pint.Quantity:
    p = to_quantity(k["p"])
    return np.stack([p.min(), p.max()])


def trange(k: xr.DataArray) -> pint.Quantity:
    t = to_quantity(k["t"])
    return np.stack([t.min(), t.max()])


def check_missing_x_coords(x_ds: list[str], x_thermoprops: list[str]):
    if not set(x_thermoprops).issubset(set(x_ds)):
        msg = (
            "The absorption dataset does not contain all the volume fraction "
            "coordinates present in the thermophysical properties dataset."
        )
        msg += f"\nVolume fraction coordinates in absorption data set: {x_ds}"

        msg += (
            f"\nVolume fraction coordinates in thermophysical properties data "
            f"set: {x_thermoprops}"
        )
        msg += (
            f"\nMissing volume fraction coordinates: "
            f"{list(set(x_thermoprops) - set(x_ds))}"
        )
        raise MissingCoordinateError(msg)


def check_scalar_x_coords(
    x_absorption_scalar: list[str],
    thermoprops: xr.Dataset,
    rtol=1e-3,
):
    x_thermoprops = [f"x_{m}" for m in thermoprops.joseki.molecules]
    if set(x_absorption_scalar).issubset(set(x_thermoprops)):
        x_intersection = set(x_absorption_scalar).intersection(set(x_thermoprops))

        _x_nonconstant = []
        for _x in x_intersection:
            values = thermoprops[_x].values
            if not np.all(np.isclose(values, values[0], rtol=rtol)):
                _x_nonconstant.append(_x)

        if _x_nonconstant:
            msg = (
                f"These volume fraction coordinates show more than {rtol:.2%} "
                f"variation in the atmosphere's thermophysical properties "
                f"but are scalar coordinates in the absorption dataset: "
                f"{_x_nonconstant}"
            )
            raise ScalarCoordinateError(msg)


def check_x_ranges(k: xr.DataArray, thermoprops: xr.Dataset) -> None:
    xrange_ds = xrange(k)
    out_of_bounds = {}
    for c in xrange_ds:
        x_thermoprops = to_quantity(thermoprops[c])
        xrange_t = np.stack([x_thermoprops.min(), x_thermoprops.max()])
        if xrange_t[0] < xrange_ds[c][0] or xrange_t[1] > xrange_ds[c][1]:
            out_of_bounds[c] = {
                "thermoprops": xrange_t,
                "absorption": xrange_ds[c],
            }

    if out_of_bounds:
        msg = "The following volume fraction coordinates are out of bounds."
        for c in out_of_bounds:
            msg += f"\n{c}: range in thermoprops {out_of_bounds[c]['thermoprops']}, "
            msg += f"range in absorption dataset {out_of_bounds[c]['absorption']}"
        raise OutOfBoundsCoordinateError(msg)


def interp_x(
    k: xr.DataArray,
    thermoprops: xr.Dataset,
    error_handler_config: dict[str, str] = {
        "missing": "ignore",
        "scalar": "ignore",
        "bounds": "raise",
    },
) -> xr.DataArray:
    """
    Interpolate absorption coefficient data along volume fraction dimensions.

    Parameters
    ----------
    k : xr.DataArray
        Absorption coefficient data array.

    thermoprops : xr.Dataset
        Atmosphere's thermophysical properties.

    error_handler_config : dict[str, str], optional
        Configuration of the error handler, by default
        DEFAULT_X_ERROR_HANDLER_CONFIG

    Returns
    -------
    xr.DataArray
        Interpolated absorption coefficient data array.
    """
    # update missing keys in handle_config with DEFAULT_HANDLE_CONFIG
    error_handler_config = {**DEFAULT_X_ERROR_HANDLER_CONFIG, **error_handler_config}

    x_thermoprops = [f"x_{m}" for m in thermoprops.joseki.molecules]
    logger.debug("x_thermoprops: %s", x_thermoprops)

    x_absorption = xcoords(k)
    x_absorption_scalar = xcoords_scalar(k)
    x_absorption_array = set(x_absorption) - set(x_absorption_scalar)
    x_absorption_interp = x_absorption_array  # interpolation along these coords
    x_absorption_sel = x_absorption_scalar  # selection along these coords
    logger.debug("x_absorption_interp: %s", x_absorption_interp)
    logger.debug("x_absorption_sel: %s", x_absorption_sel)

    # -------------------------------------------------------------------------
    #               Check coordinates for interpolation errors
    # -------------------------------------------------------------------------

    # Unspecified coordinates (always raise)  # TODO: or set this coord to zero?
    # a coordinate is present in 'k' but not in 'thermoprops'
    if not set(x_absorption).issubset(set(x_thermoprops)):
        msg = (
            f"The absorption dataset contains array coordinates ("
            f"{list(set(x_absorption) - set(x_thermoprops))} that are "
            f"not specified in the thermophysical properties dataset."
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Missing coordinates
    # a coordinate is specified in thermoprops but not in 'k'
    try:
        check_missing_x_coords(x_absorption, x_thermoprops)
    except MissingCoordinateError as e:
        handle_error(e, error_handler_config)

    # Scalar coordinates
    # a coordinate is present (and varying) in thermoprops that corresponds to
    # a scalar (one-value) coordinate in 'k'
    try:
        check_scalar_x_coords(x_absorption_scalar, thermoprops)
    except ScalarCoordinateError as e:
        handle_error(e, error_handler_config)

    # Out of bounds coordinate
    # a coordinate value in thermoprops is out of the range of the
    # corresponding coordinate in 'k'
    try:
        check_x_ranges(k, thermoprops)
        bounds_error = True
    except OutOfBoundsCoordinateError as e:
        handle_error(e, error_handler_config)
        bounds_error = False

    # -------------------------------------------------------------------------
    #                   Select and/or interpolate
    # -------------------------------------------------------------------------

    # Select
    k_selected = k.isel(**{x: 0 for x in x_absorption_sel}, drop=True)

    # Interpolate
    fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?

    k_interp = k_selected.interp(
        **{x: thermoprops[x] for x in x_absorption_interp},
        kwargs={
            "bounds_error": bounds_error,
            "fill_value": fill_value,
        },
    ).drop_vars(x_absorption_interp)

    return k_interp


def check_p_range(k: xr.Dataset, thermoprops: xr.Dataset):
    p_thermoprops = to_quantity(thermoprops.p)
    p_thermoprops_range = np.stack([p_thermoprops.min(), p_thermoprops.max()])
    p_k = to_quantity(k.p)
    p_k_range = np.stack([p_k.min(), p_k.max()])

    if p_thermoprops.min() < p_k.min() or p_thermoprops.max() > p_k.max():
        raise OutOfBoundsCoordinateError(
            f"The pressure range of the thermophysical properties dataset "
            f"({p_thermoprops_range}) is outside the pressure range of the "
            f"absorption dataset ({p_k_range})."
        )


def interp_p(
    k: xr.DataArray,
    thermoprops: xr.Dataset,
    error_handler_config: dict[str, str] = {
        "missing": "raise",
        "scalar": "raise",
        "bounds": "ignore",
    },
) -> xr.Dataset:
    """
    Interpolate absorption dataset along pressure dimension.

    Parameters
    ----------
    k : xr.DataArray
        Absorption coefficient data array.

    thermoprops : xr.Dataset
        Atmosphere's thermophysical properties.

    error_handler_config : dict[str, str], optional
        Configuration of the error handler, by default
        DEFAULT_X_ERROR_HANDLER_CONFIG

    Returns
    -------
    xr.DataArray
        Interpolated absorption coefficient data array.
    """
    try:
        check_p_range(k, thermoprops)
        bounds_error = True
    except OutOfBoundsCoordinateError as e:
        handle_error(e, error_handler_config)
        bounds_error = False

    fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?

    k_interp = k.interp(
        p=thermoprops.p,
        kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
    ).drop_vars("p")

    return k_interp


def check_t_range(k: xr.DataArray, thermoprops: xr.Dataset):
    t_thermoprops = to_quantity(thermoprops.t)
    t_thermoprops_range = np.stack([t_thermoprops.min(), t_thermoprops.max()])
    t_ds = to_quantity(k.t)
    t_ds_range = np.stack([t_ds.min(), t_ds.max()])

    if t_thermoprops.min() < t_ds.min() or t_thermoprops.max() > t_ds.max():
        raise OutOfBoundsCoordinateError(
            f"The temperature range of the thermophysical properties dataset "
            f"({t_thermoprops_range}) is outside the temperature range of the "
            f"absorption dataset ({t_ds_range})."
        )


def interp_t(
    k: xr.DataArray,
    thermoprops: xr.Dataset,
    error_handler_config: dict[str, str] = {
        "missing": "raise",
        "scalar": "raise",
        "bounds": "ignore",
    },
) -> xr.Dataset:
    """
    Interpolate absorption coefficient data set along temperature dimension.

    Parameters
    ----------
    k : xr.DataArray
        Absorption coefficient data array.

    thermoprops : xr.Dataset
        Atmosphere's thermophysical properties.

    error_handler_config : dict[str, str], optional
        Configuration of the error handler, by default
        DEFAULT_X_ERROR_HANDLER_CONFIG

    Returns
    -------
    xr.DataArray
        Interpolated absorption coefficient data array.
    """
    try:
        check_t_range(k, thermoprops)
        bounds_error = True
    except OutOfBoundsCoordinateError as e:
        handle_error(e, error_handler_config)
        bounds_error = False

    fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?

    k_interp = k.interp(
        t=thermoprops.t,
        kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
    ).drop_vars("t")

    return k_interp


# ------------------------------------------------------------------------------
#       Absorption coefficient evaluation implementation in mono modes
# ------------------------------------------------------------------------------


def wrange_mono(ds: xr.Dataset) -> P.Interval[pint.Quantity]:
    wds = to_quantity(ds.w)
    wunits = ucc.get("wavelength")

    if wds.check("[length]^-1"):
        return P.closed(
            (1 / wds.max()).to(wunits),
            (1 / wds.min()).to(wunits),
        )
    elif wds.check("[length]"):
        return P.closed(
            wds.min().to(wunits),
            wds.max().to(wunits),
        )
    else:
        raise ValueError(
            f"Spectral coordinate of absorption dataset has unexpected units "
            f"({ds.w.units})."
        )


def check_w_range_mono(ds: xr.Dataset, w: pint.Quantity):
    wrange = wrange_mono(ds=ds)
    wmin, wmax = wrange.lower, wrange.upper

    if np.any((w < wmin) | (w > wmax)):
        msg = (
            f"Requested w coordinate ({w}) is outside the range of the "
            f"absorption data set ({wmin} to {wmax})."
        )
        raise OutOfBoundsCoordinateError(msg)


def interp_w(ds: xr.Dataset, w: pint.Quantity) -> xr.DataArray:
    """
    Interpolate absorption data set to a given wavelength.

    Parameters
    ----------
    ds : xr.Dataset
        Absorption data set.

    w : pint.Quantity
        Wavelength.

    Returns
    -------
    xr.DataArray
        Wavelength-interpolated absorption data.
    """
    # convert to wavenumber/wavelength
    wunits = ds.w.units
    if ureg(wunits).check("[length]^-1"):
        # interpolation according to wavenumber
        wm = (1 / w).m_as(wunits)
    elif ureg(wunits).check("[length]"):
        # interpolation according to wavelength
        wm = w.m_as(wunits)
    else:
        raise ValueError(f"Cannot interpret units {wunits}")

    # check wavelength range and interpolate
    check_w_range_mono(ds, w)
    return ds.sigma_a.interp(
        w=wm,
        method="linear",
    )


def eval_sigma_a_mono_impl_ds(
    ds: xr.Dataset,
    thermoprops: xr.Dataset,
    w: pint.Quantity,
    error_handler_config: dict[str, dict[str, str]] | None = None,
) -> pint.Quantity:
    """
    Evaluate absorption coefficient in given spectral and thermophysical
    conditions.

    Parameters
    ----------
    ds : Dataset
        Absorption coefficient dataset.

    thermoprops : Dataset
        Atmospheric thermophysical properties data set.

    w : quantity
        Wavelength.

    error_handler_config : dict
        Error handler configuration.

    Returns
    -------
    quantity
        Absorption coefficient values.
    """
    if error_handler_config is None:
        error_handler_config = DEFAULT_HANDLER_CONFIG

    kw = interp_w(ds, w)
    kwx = interp_x(kw, thermoprops, error_handler_config["x"])
    kwxp = interp_p(kwx, thermoprops, error_handler_config["p"])
    kwxpt = interp_t(kwxp, thermoprops, error_handler_config["t"])
    return kwxpt


def eval_sigma_a_mono_impl(
    absorption_data: dict[P.Interval, xr.Dataset],
    thermoprops: xr.Dataset,
    w: pint.Quantity,
    error_handler_config: dict[str, dict[str, str]] | None = None,
) -> pint.Quantity:
    # NOTE: it is assumed that wavelength intervals do no intersect, namely that there
    # is at most one dataset relevant to compute the absorption coefficient at
    # a given wavelength
    # TODO: check that in absorption_data validator

    # NOTE: if the evaluation wavelengths span only one dataset, the evaluation
    # is vectorized. Otherwise, the evaluation is vectorized only when each
    # individual dataset is evaluated.

    if error_handler_config is None:
        error_handler_config = DEFAULT_HANDLER_CONFIG

    w = np.atleast_1d(w)
    wargsort = np.argsort(w)
    ws = w[wargsort]
    wargunsort = np.argsort(wargsort)
    # assume that w might be a large array

    das = []
    upper_max = max([i.upper for i in absorption_data.keys()])
    for interval, dataset in absorption_data.items():
        # we assume that this list is not too long (<10 elements) so the
        # performance cost of this loop is not too high.

        # find the wavelength that are contained in the current wavelength
        # interval:
        if interval.upper != upper_max:
            w_where = ws[
                (ws >= interval.lower)
                & (ws < interval.upper)  # mind the strict inequality
            ]
        else:
            w_where = ws[(ws >= interval.lower) & (ws <= interval.upper)]

        if w_where.size > 0:
            da = eval_sigma_a_mono_impl_ds(
                ds=dataset,
                thermoprops=thermoprops,
                w=w_where,
                error_handler_config=error_handler_config,
            )
            das.append(da)

    concatenated = xr.concat(das, dim="w")
    return to_quantity(concatenated.isel(w=wargunsort).transpose("w", "z"))


# ------------------------------------------------------------------------------
#       Absorption coefficient evaluation implementation in CKD modes
# ------------------------------------------------------------------------------


def wrange_ckd(ds: xr.Dataset) -> P.Interval[pint.Quantity]:
    wbounds = to_quantity(ds.wbounds.squeeze())
    wunits = ucc.get("wavelength")
    if wbounds.check("[length]^-1"):
        return P.closed(tuple(np.sort((1 / wbounds.to(wunits)))))
    elif wbounds.check("[length]"):
        return P.closed(*tuple(np.sort(wbounds).to(wunits)))
    else:
        raise ValueError(
            f"Spectral coordinate of absorption dataset has unexpected units "
            f"({ds.w.units})."
        )


def check_w_range_ckd(ds: xr.Dataset, w: pint.Quantity, rtol: float = 1e-3):
    # This function is both for CKD absorption datasets

    # Check that the evaluation wavelength is amongst the absorption dataset
    # spectral coordinate values (to a relative tolerance of 'rtol').

    wds = to_quantity(ds.w)
    if wds.check("[length]^-1"):
        w_k = (1 / wds).to(w.units)
    elif wds.check("[length]"):
        w_k = wds
    else:
        raise ValueError(
            f"Spectral coordinate of absorption dataset has unexpected units "
            f"({ds.w.units})."
        )

    if not np.any(np.isclose(w_k, w, rtol=rtol)):
        msg = (
            f"Requested w coordinate ({w}) is not amongst the spectral "
            f"coordinates of the absorption data set ({w_k})."
        )
        raise OutOfBoundsCoordinateError(msg)


def interp_wg(ds: xr.Dataset, w: pint.Quantity, g: float) -> xr.DataArray:
    """
    Interpolate absorption data set to given w and g coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Absorption data set.

    w : pint.Quantity
        Wavelength corresponding to the CKD bin to select.

    g : float
        g-point to interpolate at.

    Returns
    -------
    xr.DataArray
        Spectral-interpolated absorption data.
    """
    # 1. bin selection
    wunits = ds.w.units
    if ureg(wunits).check("[length]^-1"):
        # select bin according to wavenumber
        wm = (1 / w).m_as(wunits)
    elif ureg(wunits).check("[length]"):
        # select bin according to wavelength
        wm = w.m_as(wunits)
    else:
        raise ValueError(f"Cannot interpret units {wunits}")

    check_w_range_ckd(ds, w, rtol=1e-3)
    # select the bin
    kw = ds.sigma_a.sel(
        w=wm,
        method="nearest",
    ).expand_dims("w")

    # 2. interpolation along g-point (pseudo-spectral coordinate)
    return kw.interp(g=g).drop_vars("g")


def eval_sigma_a_ckd_impl_ds(
    ds: xr.Dataset,
    thermoprops: xr.Dataset,
    w: pint.Quantity,
    g: float,
    error_handler_config: dict[str, dict[str, str]] | None = None,
) -> pint.Quantity:
    """
    Evaluate absorption coefficient in given spectral and thermophysical
    conditions.

    Parameters
    ----------
    ds : Dataset
        Absorption coefficient dataset.

    thermoprops : Dataset
        Atmospheric thermophysical properties data set.

    w : quantity
        Wavelength.

    g: float
        g-point.

    error_handler_config : dict
        Error handler configuration.

    Returns
    -------
    quantity
        Absorption coefficient values.
    """
    if error_handler_config is None:
        error_handler_config = DEFAULT_HANDLER_CONFIG

    kw = interp_wg(ds, w, g)
    kwx = interp_x(kw, thermoprops, error_handler_config["x"])
    kwxp = interp_p(kwx, thermoprops, error_handler_config["p"])
    kwxpt = interp_t(kwxp, thermoprops, error_handler_config["t"])
    return kwxpt


def _isclose(a, b, **kwargs):
    """Return values in b that are close to one value in a (different
    shapes supported)."""
    out = []
    for value in a:
        isclosed = b[np.isclose(b, value, **kwargs)]
        if len(isclosed) > 0:
            out.append(isclosed)
    if len(out) > 0:
        return np.stack(out).squeeze()
    else:
        return np.empty(0)


def w_dataset(ds: xr.Dataset) -> pint.Quantity:
    w = to_quantity(ds.w)
    wunits = ucc.get("wavelength")
    if w.check("[length]"):
        return w.to(wunits)
    elif w.check("[length^-1]"):
        return (1 / w).to(wunits)
    else:
        raise ValueError


def eval_sigma_a_ckd_impl(
    absorption_data,
    thermoprops: xr.Dataset,
    w: pint.Quantity,
    g: float,
    error_handler_config: dict[str, dict[str, str]] | None = None,
) -> pint.Quantity:
    # NOTE: it is assumed that only one dataset corresponds to each wavelength
    # value in 'w'

    if error_handler_config is None:
        error_handler_config = DEFAULT_HANDLER_CONFIG

    w = np.atleast_1d(w)
    wargsort = np.argsort(w)
    ws = w[wargsort]
    wargunsort = np.argsort(wargsort)

    das = []
    for _, dataset in absorption_data.items():
        # we assume that this list is not too long (~10 elements) so the
        # performance cost of this loop is not too high.

        # find the wavelength values that match with the current dataset
        w_ds = w_dataset(dataset)
        w_where = _isclose(ws, w_ds, rtol=1e-3)

        if w_where.size > 0:
            da = eval_sigma_a_ckd_impl_ds(
                ds=dataset,
                thermoprops=thermoprops,
                w=w_where,
                g=g,
                error_handler_config=error_handler_config,
            )
            das.append(da)

    if len(das) == 0:
        w_dss = []
        w_wheres = []
        for _, dataset in absorption_data.items():
            w_ds = w_dataset(dataset)
            w_where = _isclose(ws, w_ds, rtol=1e-3)
            w_dss.append(w_ds)
            w_wheres.append(w_where)

        raise ValueError(f"{w_dss=}, {w_wheres=}")

    concatenated = xr.concat(das, dim="w")
    return to_quantity(concatenated.isel(w=wargunsort).transpose("w", "z"))


# ------------------------------------------------------------------------------
#                wavelength range based on active mode
# ------------------------------------------------------------------------------


def wrange(ds: xr.Dataset):
    if eradiate.mode().is_mono:
        wrange = wrange_mono(ds)
    elif eradiate.mode().is_ckd:
        wrange = wrange_ckd(ds)
    else:
        raise UnsupportedModeError
    return wrange
