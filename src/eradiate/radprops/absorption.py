"""
Functions to compute monochromatic absorption.
"""
from __future__ import annotations

import numpy as np
import pint
import xarray as xr
from scipy.constants import physical_constants

from ..units import to_quantity
from ..units import unit_registry as ureg

_BOLTZMANN = ureg.Quantity(*physical_constants["Boltzmann constant"][:2])


def compute_sigma_a(
    ds: xr.Dataset,
    wl: pint.Quantity = ureg.Quantity(550.0, "nm"),
    p: pint.Quantity = ureg.Quantity(101325.0, "Pa"),
    t: pint.Quantity = ureg.Quantity(288.15, "K"),
    n: pint.Quantity | None = None,
    fill_values: float | None = None,
    methods: dict[str, str] | None = None,
) -> pint.Quantity:
    """
    Compute monochromatic absorption coefficient at given wavelength,
    pressure and temperature values.

    Parameters
    ----------
    ds : Dataset
        Absorption cross-section data set.

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
    The values of the absorption cross-section at the desired wavelength,
    pressure and temperature values,
    :math:`\\sigma_{a\\lambda} (p, T)`,
    are obtained by interpolating the input absorption cross-section data
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
