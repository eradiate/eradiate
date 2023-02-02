"""
Functions to compute monochromatic absorption.
"""
from __future__ import annotations

import enum
import logging
import typing as t
import warnings

import numpy as np
import pint
import xarray as xr
from scipy.constants import physical_constants

from ..exceptions import (
    CoordinateRangeError,
    InterpolationError,
    MissingCoordinateError,
    ScalarCoordinateError,
)
from ..units import to_quantity
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)

_BOLTZMANN = ureg.Quantity(*physical_constants["Boltzmann constant"][:2])


class InterpolationErrorHandler:
    def __init__(self, action: Action):
        self.action = action

    def __call__(self, e: InterpolationError):
        if self.action == Action.RAISE:
            raise e
        elif self.action == Action.WARN:
            warnings.warn(str(e), UserWarning)
        elif self.action == Action.IGNORE:
            pass
        else:
            raise NotImplementedError(f"Action {self.action} not implemented.")


INTERPOLATION_ERROR_ALIAS = {
    MissingCoordinateError: "missing",
    ScalarCoordinateError: "scalar",
    CoordinateRangeError: "range",
}

DEFAULT_X_ERROR_HANDLER_CONFIG = {
    "missing": "raise",  # the missing molecule could be a strong absorber
    "scalar": "warn",  # there are too many molecules for absorption dataset
    "range": "raise",
}

DEFAULT_P_ERROR_HANDLER_CONFIG = {
    "missing": "raise",
    "scalar": "raise",
    "range": "warn",  # thermoprops.p easily goes very low, below ds.p.min()
}

DEFAULT_T_ERROR_HANDLER_CONFIG = {
    "missing": "raise",
    "scalar": "raise",
    "range": "warn",  # thermoprops.t easily goes beyond ds.t range
}


class Action(enum.Enum):

    RAISE = enum.auto()
    WARN = enum.auto()
    IGNORE = enum.auto()

    @classmethod
    def from_str(cls, s: str) -> Action:
        return cls[s.upper()]


def xcoords(ds: xr.Dataset) -> t.List[str]:
    return [c for c in ds.coords if c.startswith("x_")]


def xrange(ds: xr.Dataset) -> t.Mapping[str, pint.Quantity]:
    _xrange = {}
    for c in xcoords(ds):
        if ds[c].size > 1:
            x = to_quantity(ds[c])
            _xrange[c] = np.stack([x.min(), x.max()])
    return _xrange


def prange(ds: xr.Dataset) -> t.Mapping[str, pint.Quantity]:
    p = to_quantity(ds["p"])
    return np.stack([p.min(), p.max()])


def trange(ds: xr.Dataset) -> t.Mapping[str, pint.Quantity]:
    t = to_quantity(ds["t"])
    return np.stack([t.min(), t.max()])


def check_w_range(ds: xr.Dataset, w: pint.Quantity) -> None:
    w_ds = to_quantity(ds.w)
    wmin = w_ds.min()
    wmax = w_ds.max()
    if np.any((w < wmin) | (w > wmax)):
        msg = (
            f"Requested w coordinate ({w}) is outside the range of the "
            f"absorption data set ({wmin} to {wmax})."
        )
        raise CoordinateRangeError(msg)


def interp_w(ds: xr.Dataset, w: pint.Quantity) -> xr.Dataset:
    """
    Interpolate absorption data set to a given w coordinate.
    """

    check_w_range(ds, w)  # out-of-range w coordinate is a fatal error

    wunits = ds.w.attrs["units"]
    return ds.interp(w=w.m_as(wunits))


def check_missing_x_coords(x_ds: t.List[str], x_thermoprops: t.List[str]) -> None:
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
    x_ds_non_scalar: t.List[str],
    x_thermoprops: t.List[str],
) -> None:
    if not set(x_thermoprops).issubset(set(x_ds_non_scalar)):
        msg = (
            "Some volume fraction coordinates in the thermophysical properties "
            "dataset correspond to scalar (one-value) coordinate in the "
            "absorption data set."
        )
        msg += (
            f"\nNon scalar volume fraction coordinates in absorption data "
            f"set: {x_ds_non_scalar}"
        )

        msg += (
            f"\nVolume fraction coordinates in thermophysical properties data "
            f"set: {x_thermoprops}"
        )
        msg += (
            f"\nThese volume fraction coordinates are scalar: "
            f"{list(set(x_thermoprops) - set(x_ds_non_scalar))}"
        )
        raise ScalarCoordinateError(msg)


def check_x_ranges(ds: xr.Dataset, thermoprops: xr.Dataset) -> None:
    xrange_ds = xrange(ds)
    out_of_bound = {}
    for c in xrange_ds:
        x_thermoprops = to_quantity(thermoprops[c])
        xrange_t = np.stack([x_thermoprops.min(), x_thermoprops.max()])
        if xrange_t[0] < xrange_ds[c][0] or xrange_t[1] > xrange_ds[c][1]:
            out_of_bound[c] = {
                "thermoprops": xrange_t,
                "absorption": xrange_ds[c],
            }

    if out_of_bound:
        msg = "The following volume fraction coordinates are out of bound."
        for c in out_of_bound:
            msg += f"\n{c}: range in thermoprops {out_of_bound[c]['thermoprops']}, "
            msg += f"range in absorption dataset {xrange_ds[c]['absorption']}"
        raise CoordinateRangeError(msg)


def handle_error(e: InterpolationError, config: t.Mapping[str, str]):
    error_alias = INTERPOLATION_ERROR_ALIAS[type(e)]
    action = Action.from_str(config[error_alias])
    handler = InterpolationErrorHandler(action)
    handler(e)


def interp_x(
    ds: xr.Dataset,
    thermoprops: xr.Dataset,
    error_handler_config: t.Mapping[str, str] = DEFAULT_X_ERROR_HANDLER_CONFIG,
) -> xr.Dataset:
    """
    Interpolate absorption dataset along volume fraction dimensions.
    """
    # update missing keys in handle_config with DEFAULT_HANDLE_CONFIG
    error_handler_config = {**DEFAULT_X_ERROR_HANDLER_CONFIG, **error_handler_config}

    x_thermoprops = [dv for dv in thermoprops.data_vars if dv.startswith("x_")]
    logger.debug("x_thermoprops: %s", x_thermoprops)

    x_ds = xcoords(ds)
    x_ds_interp = list(xrange(ds).keys())  # ds will be interpolated along these coords
    x_ds_sel = set(x_ds) - set(x_ds_interp)  # ds will be selected along these coords
    logger.debug("x_ds_interp: %s", x_ds_interp)
    logger.debug("x_ds_sel: %s", x_ds_sel)

    # -------------------------------------------------------------------------
    # Check coordinates for interpolation errors
    # -------------------------------------------------------------------------

    # Unspecified coordinates (alyways raise)
    # a nonscalar coordinate is present in ds but not in thermoprops
    if not set(x_ds_interp).issubset(set(x_thermoprops)):
        raise ValueError(
            f"The absorption dataset contains coordinates ("
            f"{list(set(x_ds_interp)-set(x_thermoprops))} that are not "
            f"specified in the thermophysical properties dataset."
        )

    # Missing coordinates
    # a coordinate is specified in thermoprops but not in ds
    try:
        check_missing_x_coords(x_ds, x_thermoprops)
    except MissingCoordinateError as e:
        handle_error(e, error_handler_config)

    # Scalar coordinates
    # a coordinate is present in thermoprops that corresponds to a scalar
    # (one-value) coordinate in ds
    try:
        check_scalar_x_coords(x_ds_interp, x_thermoprops)
    except ScalarCoordinateError as e:
        handle_error(e, error_handler_config)

    # Coordinate ranges
    # a coordinate value in thermoprops is out of the range of the
    # corresponding coordinate in ds
    try:
        check_x_ranges(ds, thermoprops)
        bounds_error = True
    except CoordinateRangeError as e:
        handle_error(e, error_handler_config)
        bounds_error = False

    # -------------------------------------------------------------------------
    # Select / Interpolate
    # -------------------------------------------------------------------------

    # Select
    ds = ds.isel(**{x: 0 for x in x_ds_sel})

    # Interpolate
    fill_value = None if bounds_error else (ds.k.min(), ds.k.max())

    ds = ds.interp(
        **{x: thermoprops[x] for x in x_ds_interp},
        kwargs={
            "bounds_error": bounds_error,
            "fill_value": fill_value,
        },
    )

    return ds


def check_p_range(ds: xr.Dataset, thermoprops: xr.Dataset):
    p_thermoprops = to_quantity(thermoprops.p)
    p_range = np.stack([p_thermoprops.min(), p_thermoprops.max()])
    p_ds = to_quantity(ds.p)
    p_ds_range = np.stack([p_ds.min(), p_ds.max()])

    if p_thermoprops.min() < p_ds.min() or p_thermoprops.max() > p_ds.max():
        raise CoordinateRangeError(
            f"The pressure range of the thermophysical properties dataset "
            f"({p_range}) is outside the pressure range of the absorption "
            f"dataset ({p_ds_range})."
        )


def interp_p(
    ds: xr.Dataset,
    thermoprops: xr.Dataset,
    error_handler_config: t.Mapping[str, str] = DEFAULT_P_ERROR_HANDLER_CONFIG,
) -> xr.Dataset:
    """
    Interpolate absorption dataset along pressure dimension.
    """

    try:
        check_p_range(ds, thermoprops)
        bounds_error = True
    except CoordinateRangeError as e:
        handle_error(e, error_handler_config)
        bounds_error = False

    fill_value = None if bounds_error else (ds.k.min(), ds.k.max())

    ds = ds.interp(
        p=thermoprops.p,
        kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
    )

    return ds


def check_t_range(ds: xr.Dataset, thermoprops: xr.Dataset):
    t_thermoprops = to_quantity(thermoprops.t)
    t_range = np.stack([t_thermoprops.min(), t_thermoprops.max()])
    t_ds = to_quantity(ds.t)
    t_ds_range = np.stack([t_ds.min(), t_ds.max()])

    if t_thermoprops.min() < t_ds.min() or t_thermoprops.max() > t_ds.max():
        raise CoordinateRangeError(
            f"The temperature range of the thermophysical properties dataset "
            f"({t_range}) is outside the temperature range of the absorption "
            f"dataset ({t_ds_range})."
        )


def interp_t(
    ds: xr.Dataset,
    thermoprops: xr.Dataset,
    error_handler_config: t.Mapping[str, str] = DEFAULT_T_ERROR_HANDLER_CONFIG,
) -> xr.Dataset:
    """
    Interpolate absorption coefficient data set along temperature dimension.
    """
    try:
        check_t_range(ds, thermoprops)
        bounds_error = True
    except CoordinateRangeError as e:
        handle_error(e, error_handler_config)
        bounds_error = False

    fill_value = None if bounds_error else (ds.k.min(), ds.k.max())

    ds = ds.interp(
        t=thermoprops.t,
        kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
    )

    return ds


def compute(
    ds: xr.Dataset,
    thermoprops: xr.Dataset,
    w: pint.Quantity = 550.0 * ureg.nm,
    x_error_handler_config: t.Mapping[str, str] = DEFAULT_X_ERROR_HANDLER_CONFIG,
    p_error_handler_config: t.Mapping[str, str] = DEFAULT_P_ERROR_HANDLER_CONFIG,
    t_error_handler_config: t.Mapping[str, str] = DEFAULT_T_ERROR_HANDLER_CONFIG,
) -> pint.Quantity:
    """
    Compute absorption coefficient in a given atmospheric profile.

    Parameters
    ----------
    ds : Dataset
        Absorption coefficient data set.

    thermoprops : Dataset
        Atmospheric thermophysical properties data set.

    w : quantity
        Wavelength (scalar or array).

    Returns
    -------
    quantity
        Absorption coefficient values.

    Notes
    -----

    """
    # interpolate first along wavelength because this is usually the largest
    # dimension
    ds = interp_w(ds, w)

    # then interpolate along volume fraction, pressure and temperature
    ds = interp_x(ds, thermoprops, x_error_handler_config)
    ds = interp_p(ds, thermoprops, p_error_handler_config)
    ds = interp_t(ds, thermoprops, t_error_handler_config)

    k = ds.k.squeeze()
    return to_quantity(k)


def compute_sigma_a(
    ds: xr.Dataset,
    wl: pint.Quantity = ureg.Quantity(550.0, "nm"),
    p: pint.Quantity = ureg.Quantity(101325.0, "Pa"),
    t: pint.Quantity = ureg.Quantity(288.15, "K"),
    n: t.Optional[pint.Quantity] = None,
    fill_values: t.Optional[float] = None,
    methods: t.Optional[t.Dict[str, str]] = None,
) -> pint.Quantity:
    """
    Compute monochromatic absorption coefficient at given wavelength,
    pressure and temperature values.

    Parameters
    ----------
    ds : Dataset
        Absorption cross section data set.

    wl : quantity
        Wavelength [nm].

    p : quantity
        Pressure [Pa].

        .. note:: If ``p``, ``t`` and ``n`` are arrays, their lengths must be
           the same.

    t : quantity
        Temperature [K].

        .. note:: If the coordinate ``t`` is not in the input dataset ``ds``,
           the interpolation on temperature is not performed.

    n : quantity
        Number density [m^-3].

        .. note:: If ``n`` is ``None``, the values of ``t`` and ``p`` are then
           used only to compute the corresponding number density.

    fill_values : dict, optional
        Mapping of coordinates (in ``["w", "pt"]``) and fill values (either
        ``None`` or float).
        If not ``None``, out of bounds values are assigned the fill value
        during interpolation along the wavelength or pressure and temperature
        coordinates.
        If ``None``, out of bounds values will trigger the raise of a
        ``ValueError``.
        Only one fill value can be provided for both pressure and temperature
        coordinates.

    methods : dict, optional
        Mapping of coordinates (in ``["w", "pt"]``) and interpolation methods.
        Default interpolation method is linear.
        Only one interpolation method can be specified for both pressure
        and temperature coordinates.

    Returns
    -------
    quantity
        Absorption coefficient values.

    Raises
    ------
    ValueError
        When wavelength, pressure, or temperature values are out of the range
        of the data set and the corresponding fill value in ``fill_values`` is
        ``None``.

    Warnings
    --------
    The values of the absorption cross section at the desired wavelength,
    pressure and temperature values,
    :math:`\\sigma_{a\\lambda} (p, T)`,
    are obtained by interpolating the input absorption cross section data
    set along the corresponding dimensions.

    Notes
    -----
    The absorption coefficient is given by:

    .. math::
        k_{a\\lambda} = n \\, \\sigma_{a\\lambda} (p, T)

    where

    * :math:`k_{a\\lambda}` is the absorption coefficient [:math:`L^{-1}`],
    * :math:`\\lambda` is the wavelength [:math:`L`],
    * :math:`n` is the number density [:math:`L^{-3}`],
    * :math:`\\sigma_a` is the absorption cross section [:math:`L^2`],
    * :math:`p` is the pressure [:math:`ML^{-1}T^{-2}`] and
    * :math:`t` is the temperature [:math:`\\Theta`].
    """

    if fill_values is None:
        fill_values = dict(w=None, pt=None)

    if methods is None:
        methods = dict(w="linear", pt="linear")

    for name in ["w", "pt"]:
        if name not in fill_values:
            fill_values[name] = None
        if name not in methods:
            methods[name] = "linear"

    # Interpolate along wavenumber dimension
    xsw = ds.interp(
        w=(1.0 / wl).m_as(ds.w.units),  # wavenumber in cm^-1
        method=methods["w"],
        kwargs=dict(
            bounds_error=(fill_values["w"] is None),
            fill_value=fill_values["w"],
        ),
    )

    # If the data set includes a temperature coordinate, we interpolate along
    # both pressure and temperature dimensions.
    # Else, we interpolate only along the pressure dimension.
    p_m = p.m_as(ds.p.units)
    p_values = np.array([p_m]) if isinstance(p_m, float) else p_m
    pz = xr.DataArray(p_values, dims="pt")
    if "t" in ds.coords:
        t_m = t.m_as(ds.t.units)
        t_values = np.array([t_m] * len(p_values)) if isinstance(t_m, float) else t_m
        tz = xr.DataArray(t_values, dims="pt")
        interpolated = xsw.interp(
            p=pz,
            t=tz,
            method=methods["pt"],
            kwargs=dict(
                bounds_error=(fill_values["pt"] is None),
                fill_value=fill_values["pt"],
            ),
        )
    else:
        interpolated = xsw.interp(
            p=pz,
            method=methods["pt"],
            kwargs=dict(
                bounds_error=(fill_values["pt"] is None),
                fill_value=fill_values["pt"],
            ),
        )

    xs = to_quantity(interpolated.xs)

    n = p / (_BOLTZMANN * t) if n is None else n

    return (n * xs).to("km^-1")
